import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import pytorch_lightning as pl
import random
import numpy as np
from omegaconf import OmegaConf

from cotatron import Cotatron
from modules import VCDecoder, ResidualEncoder
from datasets import TextMelDataset, text_mel_collate


class Synthesizer(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams  # used for pl
        hp_global = OmegaConf.load(hparams.config[0])
        hp_vc = OmegaConf.load(hparams.config[1])
        hp = OmegaConf.merge(hp_global, hp_vc)
        self.hp = hp

        self.num_speakers = len(self.hp.data.speakers)
        self.cotatron = Cotatron(hparams)
        self.residual_encoder = ResidualEncoder(hp)
        self.decoder = VCDecoder(hp)
        self.speaker = nn.Embedding(self.num_speakers, hp.chn.speaker.token)

        self.is_val_first = True

    def load_cotatron(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.cotatron.load_state_dict(checkpoint['state_dict'])

        self.cotatron.eval()
        self.cotatron.freeze()

    # this is called after validation/test loop is finished.
    # see the order of hooks being called at :
    # https://pytorch-lightning.readthedocs.io/en/latest/hooks.html
    def train(self, mode=True):
        self.training = mode
        for module in self.children():
            module.train(mode)

        self.cotatron.eval()
        self.cotatron.freeze()
        return self

    def forward(self, text, mel_source, input_lengths, output_lengths, max_input_len):
        z_s_aligner = self.cotatron.speaker(mel_source, output_lengths)
        text_encoding = self.cotatron.encoder(text, input_lengths)
        z_s_repeated = z_s_aligner.unsqueeze(1).expand(-1, text_encoding.size(1), -1)
        decoder_input = torch.cat((text_encoding, z_s_repeated), dim=2)
        _, _, alignment = \
            self.cotatron.decoder(mel_source, decoder_input, input_lengths, output_lengths, max_input_len,
                                  prenet_dropout=0.0, tfrate=False)

        # alignment: [B, T_dec, T_enc]
        # text_encoding: [B, T_enc, chn.encoder]
        linguistic = torch.bmm(alignment, text_encoding)  # [B, T_dec, chn.encoder]
        linguistic = linguistic.transpose(1, 2)  # [B, chn.encoder, T_dec]
        return linguistic, alignment

    def inference(self, text, mel_source, target_speaker):
        device = text.device
        in_len = torch.LongTensor([text.size(1)]).to(device)
        out_len = torch.LongTensor([mel_source.size(2)]).to(device)

        z_s = self.cotatron.speaker.inference(mel_source)
        text_encoding = self.cotatron.encoder.inference(text)
        z_s_repeated = z_s.unsqueeze(1).expand(-1, text_encoding.size(1), -1)
        decoder_input = torch.cat((text_encoding, z_s_repeated), dim=2)
        _, _, alignment = \
            self.cotatron.decoder(mel_source, decoder_input, in_len, out_len, in_len,
                                  prenet_dropout=0.0, no_mask=True, tfrate=False)
        ling_s = torch.bmm(alignment, text_encoding)
        ling_s = ling_s.transpose(1, 2)

        residual = self.residual_encoder.inference(mel_source)
        ling_s = torch.cat((ling_s, residual), dim=1)

        z_t = self.speaker(target_speaker)
        mel_s_t = self.decoder(ling_s, z_t)
        return mel_s_t, alignment, residual

    # masking convolution from GAN-TTS (arXiv:1909.11646)
    def get_cnn_mask(self, lengths):
        max_len = torch.max(lengths).item()
        ids = torch.arange(0, max_len, out=torch.cuda.LongTensor(max_len))
        mask = (ids >= lengths.unsqueeze(1))
        mask = mask.unsqueeze(1)
        return mask  # [B, 1, T], torch.bool

    def training_step(self, batch, batch_idx):
        text, mel_source, speakers, input_lengths, output_lengths, max_input_len = batch

        with torch.no_grad():
            ling_s, _ = self.forward(text, mel_source, input_lengths, output_lengths, max_input_len)

        z_s = self.speaker(speakers)
        mask = self.get_cnn_mask(output_lengths)
        residual = self.residual_encoder(mel_source, mask, output_lengths)
        ling_s = torch.cat((ling_s, residual), dim=1)  # [B, chn.encoder+chn.residual_out, T]
        mel_s_s = self.decoder(ling_s, z_s, mask)
        loss_rec = F.mse_loss(mel_s_s, mel_source)
        self.logger.log_loss(loss_rec, mode='train', step=self.global_step, name='rec')

        return {'loss': loss_rec}

    def validation_step(self, batch, batch_idx):
        text, mel_source, speakers, input_lengths, output_lengths, max_input_len = batch

        ling_s, alignment = self.forward(text, mel_source, input_lengths, output_lengths, max_input_len)

        z_s = self.speaker(speakers)
        mask = self.get_cnn_mask(output_lengths)
        residual = self.residual_encoder(mel_source, mask, output_lengths)
        ling_s = torch.cat((ling_s, residual), dim=1)  # [B, chn.encoder+chn.residual_out, T]
        mel_s_s = self.decoder(ling_s, z_s, mask)
        loss_rec = F.mse_loss(mel_s_s, mel_source)

        target_speakers = np.random.choice(self.hp.data.speakers, size=ling_s.size(0))
        z_t_index = torch.LongTensor([self.hp.data.speakers.index(x) for x in target_speakers]).cuda()
        z_t = self.speaker(z_t_index)
        mel_s_t = self.decoder(ling_s, z_t, mask)

        if self.is_val_first:
            self.is_val_first = False
            self.logger.log_figures(mel_source, mel_s_s, mel_s_t, alignment, residual, self.global_step)

        return {'loss_rec': loss_rec}

    def validation_end(self, outputs):
        loss_rec = torch.stack([x['loss_rec'] for x in outputs]).mean()
        self.logger.log_loss(loss_rec, mode='val', step=self.global_step, name='rec')

        self.is_val_first = True
        return {'val_loss': loss_rec}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            list(self.decoder.parameters()) + list(self.residual_encoder.parameters()) \
            + list(self.speaker.parameters()),
            lr=self.hp.train.adam.lr,
            weight_decay=self.hp.train.adam.weight_decay,
        )
        return optimizer

    def train_dataloader(self):
        trainset = TextMelDataset(self.hp, self.hp.data.train_dir, self.hp.data.train_meta, train=True, norm=True)
        return DataLoader(trainset, batch_size=self.hp.train.batch_size, shuffle=True,
                        num_workers=self.hp.train.num_workers,
                        collate_fn=text_mel_collate, pin_memory=True, drop_last=True)

    def val_dataloader(self):
        valset = TextMelDataset(self.hp, self.hp.data.val_dir, self.hp.data.val_meta, train=False, norm=True)
        return DataLoader(valset, batch_size=self.hp.train.batch_size, shuffle=False,
                        num_workers=self.hp.train.num_workers,
                        collate_fn=text_mel_collate, pin_memory=False, drop_last=False)
