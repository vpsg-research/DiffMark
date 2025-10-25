import os
import sys
import yaml
import imageio
import numpy as np
import torch
import ffmpeg
from os.path import splitext
from shutil import copyfileobj
from tempfile import NamedTemporaryFile
from skimage.transform import resize
from skimage import img_as_ubyte
from scipy.spatial import ConvexHull
import torch.nn.functional as F
from noise_layers.fsrt.srt.checkpoint import Checkpoint
from noise_layers.fsrt.srt.utils.visualize import draw_image_with_kp
from noise_layers.fsrt.modules.keypoint_detector import KPDetector
from noise_layers.fsrt.modules.expression_encoder import ExpressionEncoder
from noise_layers.fsrt.srt.model import FSRT
import torch.nn as nn

if sys.version_info[0] < 3:
    raise Exception(
        "You must use Python 3 or higher. Recommended version is Python 3.7"
    )


def normalize_kp(
    kp_source,
    kp_driving,
    kp_driving_initial,
    adapt_movement_scale=False,
    use_relative_movement=False,
):
    if adapt_movement_scale:
        source_area = ConvexHull(kp_source.data.cpu().numpy()).volume
        driving_area = ConvexHull(kp_driving_initial[0].data.cpu().numpy()).volume
        adapt_movement_scale = np.sqrt(source_area) / np.sqrt(driving_area)
    else:
        adapt_movement_scale = 1

    kp_new = kp_driving

    if use_relative_movement:
        kp_value_diff = kp_driving - kp_driving_initial
        kp_value_diff *= adapt_movement_scale
        kp_new = kp_value_diff + kp_source

    return kp_new


def extract_keypoints_and_expression(img, model, kp_detector, cfg, src=False):
    assert kp_detector is not None

    bs, c, h, w = img.shape
    nkp = kp_detector.num_kp
    with torch.no_grad():
        kps, latent_dict = kp_detector(img)
        heatmaps = latent_dict["heatmap"].view(
            bs, nkp, latent_dict["heatmap"].shape[-2], latent_dict["heatmap"].shape[-1]
        )
        feature_maps = latent_dict["feature_map"].view(
            bs,
            latent_dict["feature_map"].shape[-3],
            latent_dict["feature_map"].shape[-2],
            latent_dict["feature_map"].shape[-1],
        )

    if kps.shape[1] == 1:
        kps = kps.squeeze(1)

    expression_vector = model.expression_encoder(feature_maps, heatmaps)

    if src:
        expression_vector = expression_vector[None]

    return kps, expression_vector


def forward_model(
    model,
    expression_vector_src,
    keypoints_src,
    expression_vector_driv,
    keypoints_driv,
    img_src,
    idx_grids,
    cfg,
    max_num_pixels,
    z=None,
):
    render_kwargs = cfg["model"]["decoder_kwargs"]
    if len(img_src.shape) < 5:
        img_src = img_src.unsqueeze(1)
    if len(keypoints_src.shape) < 4:
        keypoints_src = keypoints_src.unsqueeze(1)

    if z is None:
        z = model.encoder(
            img_src,
            keypoints_src,
            idx_grids[:, :1].repeat(1, img_src.shape[1], 1, 1, 1),
            expression_vector=expression_vector_src,
        )

    target_pos = idx_grids[:, 1]
    target_kps = keypoints_driv

    _, height, width = target_pos.shape[:3]
    target_pos = target_pos.flatten(1, 2)

    target_kps = target_kps.unsqueeze(1).repeat(1, target_pos.shape[1], 1, 1)

    num_pixels = target_pos.shape[1]
    img = torch.zeros((target_pos.shape[0], target_pos.shape[1], 3))

    for i in range(0, num_pixels, max_num_pixels):
        img[:, i : i + max_num_pixels], extras = model.decoder(
            z.clone(),
            target_pos[:, i : i + max_num_pixels],
            target_kps[:, i : i + max_num_pixels],
            expression_vector=expression_vector_driv,
        )

    return img.view(img.shape[0], height, width, 3), z


def make_animation(
    source_image,
    driving_video,
    model,
    kp_detector,
    cfg,
    max_num_pixels,
    relative=False,
    adapt_movement_scale=False,
):
    _, y, x = np.meshgrid(
        np.zeros(2),
        np.arange(source_image.shape[-3]),
        np.arange(source_image.shape[-2]),
        indexing="ij",
    )
    idx_grids = np.stack([x, y], axis=-1).astype(np.float32)
    # Normalize
    idx_grids[..., 0] = (idx_grids[..., 0] + 0.5 - ((source_image.shape[-3]) / 2.0)) / (
        (source_image.shape[-3]) / 2.0
    )
    idx_grids[..., 1] = (idx_grids[..., 1] + 0.5 - ((source_image.shape[-2]) / 2.0)) / (
        (source_image.shape[-2]) / 2.0
    )
    idx_grids = torch.from_numpy(idx_grids).cuda().unsqueeze(0)
    z = None
    with torch.no_grad():
        predictions = []
        source = (
            torch.tensor(source_image.astype(np.float32)).permute(0, 3, 1, 2).cuda()
        )
        driving = torch.tensor(
            np.array(driving_video)[np.newaxis].astype(np.float32)
        ).permute(0, 4, 1, 2, 3)
        kp_source, expression_vector_src = extract_keypoints_and_expression(
            source.clone(), model, kp_detector, cfg, src=True
        )
        kp_driving_initial, _ = extract_keypoints_and_expression(
            driving[:, :, 0].cuda().clone(), model, kp_detector, cfg
        )

        for frame_idx in range(driving.shape[2]):
            driving_frame = driving[:, :, frame_idx].cuda()
            kp_driving, expression_vector_driv = extract_keypoints_and_expression(
                driving_frame.clone(), model, kp_detector, cfg
            )

            kp_norm = normalize_kp(
                kp_source=kp_source[0],
                kp_driving=kp_driving,
                kp_driving_initial=kp_driving_initial,
                use_relative_movement=relative,
                adapt_movement_scale=adapt_movement_scale,
            )

            out, z = forward_model(
                model,
                expression_vector_src,
                kp_source,
                expression_vector_driv,
                kp_norm,
                source.unsqueeze(0),
                idx_grids,
                cfg,
                max_num_pixels,
                z=z,
            )
        noised_image = torch.clamp(out[0], 0.0, 1.0)
        noised_image = noised_image * 2 - 1
        noised_image = noised_image.permute(2, 0, 1)
        return noised_image


def find_best_frame(source, driving, cpu=False):
    import face_alignment
    from scipy.spatial import ConvexHull

    def normalize_kp(kp):
        kp = kp - kp.mean(axis=0, keepdims=True)
        area = ConvexHull(kp[:, :2]).volume
        area = np.sqrt(area)
        kp[:, :2] = kp[:, :2] / area
        return kp

    fa = face_alignment.FaceAlignment(
        face_alignment.LandmarksType.TWO_D,
        flip_input=True,
        device="cpu" if cpu else "cuda",
    )
    kp_source = fa.get_landmarks(255 * source[0])[0]
    kp_source = normalize_kp(kp_source)
    norm = float("inf")
    frame_num = 0
    for i, image in enumerate(driving):
        kp_driving = fa.get_landmarks(255 * image)[0]
        kp_driving = normalize_kp(kp_driving)
        new_norm = (np.abs(kp_source - kp_driving) ** 2).sum()
        if new_norm < norm:
            norm = new_norm
            frame_num = i
    return frame_num



class Fsrt(nn.Module):
    def __init__(
        self,
        config_path="noise_layers/fsrt/runs/vox256/vox256.yaml",
        checkpoint_path="noise_layers/fsrt/fsrt_checkpoints/vox256.pt",
        driving_video_path="noise_layers/fsrt/examples/conan_obrien.mp4",
        device="cuda:0",
        max_num_pixels=65536,
    ):
        super(Fsrt, self).__init__()

        # Load the config file
        with open(config_path, "r") as f:
            self.cfg = yaml.load(f, Loader=yaml.CLoader)

        # Initialize keypoint detector and expression encoder
        self.kp_detector = KPDetector().to(device)
        self.kp_detector.load_state_dict(
            torch.load("noise_layers/fsrt/fsrt_checkpoints/kp_detector.pt")
        )
        self.expression_encoder = ExpressionEncoder(
            expression_size=self.cfg["model"]["expression_size"],
            in_channels=self.kp_detector.predictor.out_filters,
        )

        # Initialize model
        self.model = FSRT(
            self.cfg["model"], expression_encoder=self.expression_encoder
        ).to(device)
        self.model.eval()
        self.kp_detector.eval()

        # Set device
        self.device = device

        # Load driving video
        self.driving_video_path = driving_video_path
        reader = imageio.get_reader(self.driving_video_path)
        self.fps = reader.get_meta_data()["fps"]
        self.driving_video = []
        self.max_num_pixels = max_num_pixels

        encoder_module = self.model.encoder
        decoder_module = self.model.decoder
        expression_encoder_module = self.model.expression_encoder

        # Load the checkpoints
        checkpoint = Checkpoint(
            "./",
            device="cuda:0",
            encoder=encoder_module,
            decoder=decoder_module,
            expression_encoder=expression_encoder_module,
        )
        checkpoint.load(checkpoint_path)
        
    def forward(self, image_cover_mask):
        # print("Fsrt")
        with torch.no_grad():
            image = image_cover_mask[0]
            size = image.shape[-2:]
            image = (image + 1) / 2
            if image.shape[-2:] != (256, 256):
                image = F.interpolate(
                    image, size=(256, 256), mode="bilinear", align_corners=False
                )
            drive_image = image_cover_mask[1]
            drive_image = (drive_image + 1) / 2
            if drive_image.shape[-2:] != (256, 256):
                drive_image = F.interpolate(
                    drive_image,
                    size=(256, 256),
                    mode="bilinear",
                    align_corners=False,
                )
            noised_image = torch.zeros_like(image).to(image.device)
            length = image.shape[0]
            for i in range(length):
                driving_video = np.array(
                    [drive_image[(i + 1) % length].permute(1, 2, 0).cpu().numpy()]
                )
                source_image_tensor = (
                    image[(i) % length].unsqueeze(0).permute(0, 2, 3, 1)
                )
                source_image_array = source_image_tensor.cpu().numpy()
                predictions = make_animation(
                    source_image_array,
                    driving_video,
                    self.model,
                    self.kp_detector,
                    relative=True,
                    adapt_movement_scale=True,
                    cfg=self.cfg,
                    max_num_pixels=self.max_num_pixels,
                )
                predictions = predictions.to(image.device)
                noised_image[i] = predictions
            if noised_image.shape[-2:] != size:
                noised_image = F.interpolate(
                    noised_image,
                    size=size,
                    mode="bilinear",
                    align_corners=False,
                )
            return noised_image

