import torch
from lib import utils
from lib.model_interface import ModelInterface
from psp.loss import PSPLoss
from psp.nets import PSPEncoder


class PSPModel(ModelInterface):
    def set_networks(self):
        self.G = PSPEncoder().cuda(self.gpu).train()
        self.G.face_generator.generator.eval()

    def set_loss_collector(self):
        self._loss_collector = PSPLoss(self.args)

    def go_step(self, global_step):
        # load batch

        I_source = self.load_next_batch()
        self.dict["I_source"] = I_source

        # run G
        self.run_G()

        # update G
        loss_G = self.loss_collector.get_loss_G(self.dict)
        utils.update_net(self.opt_G, loss_G)

        # run D
        self.run_D()

        # update D
        self.loss_collector.get_loss_D(self.dict)

        # print images
        self.train_images = [
            self.dict["I_source"], 
            self.dict["I_recon"],
            ]

    def run_G(self):
        I_recon = self.G(self.dict["I_source"])
        id_recon = self.G.get_id(I_recon)
        with torch.no_grad():
            id_source = self.G.get_id(self.dict["I_source"]).detach()

        self.dict["I_recon"] = I_recon
        self.dict["id_source"] = id_source
        self.dict["id_recon"] = id_recon

    def run_D(self):
        pass

    def do_validation(self, step):
        with torch.no_grad():
            result_images = self.G(self.valid_source)
        self.valid_images = [
            self.valid_source, 
            result_images
            ]

    @property
    def loss_collector(self):
        return self._loss_collector
        