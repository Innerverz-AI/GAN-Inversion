import torch.nn.functional as F
import torch
from lib import utils
from lib.model_interface import ModelInterface
from psp.loss import PSPLoss
from psp.nets import GradualStyleEncoder
from packages import FaceGeneratorRosinality
from packages import CurrFace

class PSPModel(ModelInterface):
    def set_networks(self):
        self.E = GradualStyleEncoder().cuda(self.gpu).train()
        self.SG = FaceGeneratorRosinality(ckpt_path=self.args.stylegan_path).cuda(self.gpu).eval()
        self.CF = CurrFace().cuda(self.gpu).eval()
        self.w_avg = self.SG.get_w_avg()

    def set_loss_collector(self):
        self._loss_collector = PSPLoss(self.args)

    def go_step(self, global_step):
        # load batch

        I_source = self.load_next_batch()
        self.dict["I_source"] = I_source

        # run E
        self.run_E()

        # update E
        loss_E = self.loss_collector.get_loss_E(self.dict)
        utils.update_net(self.opt_E, loss_E)

        # run D
        # self.run_D()

        # update D
        self.loss_collector.get_loss_D(self.dict)

        # print images
        self.train_images = [
            self.dict["I_source"], 
            self.dict["I_recon"],
            ]

    def run_E(self):
        w_fake = self.E(self.dict["I_source"])
        I_recon, _ = self.SG(w_fake + self.w_avg) # 1024x1024
        id_recon = self.CF(I_recon)
        with torch.no_grad():
            id_source = self.CF(self.dict["I_source"])

        # d_adv = self.SG.discriminator(I_recon)

        self.dict["I_recon"] = F.interpolate(I_recon, (256,256))
        self.dict["id_source"] = id_source
        self.dict["id_recon"] = id_recon
        # self.dict["d_adv"] = d_adv

    def run_D(self):
        pass

    def do_validation(self, global_step):
        with torch.no_grad():
            w_fake = self.E(self.valid_source)
            result_images, _ = self.SG(w_fake + self.w_avg) # 1024x1024
        self.valid_images = [
            self.valid_source, 
            F.interpolate(result_images, (256,256))
            ]

    @property
    def loss_collector(self):
        return self._loss_collector
        