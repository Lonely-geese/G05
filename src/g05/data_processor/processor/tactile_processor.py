"""Keep tactile images out of VLM image slots and camera subsampling."""

from .galaxea_cot_processor import GalaxeaCoTProcessor


class TactileCoTProcessor(GalaxeaCoTProcessor):
    def process_images(self, data):
        return super().process_images(data, modality="visual")

    def _process_tensors(self, data):
        sample = super()._process_tensors(data)
        tactile = super().process_images(data, modality="tactile")
        if not tactile:
            raise ValueError("TactileCoTProcessor requires images with modality: tactile")
        sample["tactile_pixel_values"] = tactile
        return sample
