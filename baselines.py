import os
import torch
import torch.nn.functional as F
from PIL import Image


class ImageEmbedder:
    def __init__(self, model, preprocessor):
        """Project images to vectors and preprocess image files for the model."""
        self.model = model
        self.processor = preprocessor


# Official Checkpoints from Salesforce & ChatIR (3 Core Models)
BLIP_CKPTS = {
    'chatir': 'chatir_weights.ckpt',
    'cocoft': 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_retrieval_coco.pth',
    'zs':     'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base.pth',
}


def BLIP_BASELINE(device="cuda", ckpt=None, mode='chatir'):
    from torchvision import transforms
    from torchvision.transforms.functional import InterpolationMode
    import sys

    for p in ['./BLIP', 'BLIP', '../BLIP', './Diffusion-Agumented-Retrieval/BLIP']:
        if os.path.exists(p) and p not in sys.path:
            sys.path.insert(0, p)

    from BLIP.models.blip_itm import blip_itm

    # Determine checkpoint
    if ckpt is not None and ckpt != '':
        blip_ckpt = BLIP_CKPTS.get(ckpt, ckpt)
    elif mode == 'zs':
        blip_ckpt = BLIP_CKPTS['zs']
    elif mode == 'cocoft':
        blip_ckpt = BLIP_CKPTS['cocoft']
    else:  # mode == 'chatir'
        blip_ckpt = BLIP_CKPTS['chatir']
        if not os.path.exists(blip_ckpt):
            for alt in ['../chatir_weights.ckpt', 'Diffusion-Agumented-Retrieval/chatir_weights.ckpt']:
                if os.path.exists(alt):
                    blip_ckpt = alt
                    break
            else:
                blip_ckpt = BLIP_CKPTS['cocoft']

    med_config = 'BLIP/configs/med_config.json'
    if not os.path.exists(med_config):
        for alt in ['configs/med_config.json', '../BLIP/configs/med_config.json', 'Diffusion-Agumented-Retrieval/BLIP/configs/med_config.json']:
            if os.path.exists(alt):
                med_config = alt
                break

    print(f"Loading BLIP model: {blip_ckpt} (med_config={med_config})")
    model = blip_itm(
        pretrained=blip_ckpt,
        med_config=med_config,
        image_size=224,
        vit='base'
    )

    model = model.to(device).eval()

    transform_test = transforms.Compose([
        transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.48145466, 0.4578275, 0.40821073),
            (0.26862954, 0.26130258, 0.27577711)
        )
    ])

    def blip_project_img(image):
        embeds = model.visual_encoder(image)
        projection = model.vision_proj(embeds[:, 0, :])
        return F.normalize(projection, dim=-1)

    def blip_prep_image(path):
        raw = Image.open(path).convert('RGB')
        return transform_test(raw)

    image_embedder = ImageEmbedder(blip_project_img, lambda path: blip_prep_image(path))

    def dialog_encoder(dialog):
        text = model.tokenizer(
            dialog,
            padding='longest',
            truncation=True,
            max_length=256,
            return_tensors="pt"
        ).to(device)

        text_output = model.text_encoder(
            text.input_ids,
            attention_mask=text.attention_mask,
            return_dict=True,
            mode='text'
        )

        shift = model.text_proj(text_output.last_hidden_state[:, 0, :])
        return F.normalize(shift, dim=-1)

    return dialog_encoder, image_embedder


def BLIP_ZERO_SHOT(device="cuda", ckpt=None):
    """Pure Zero-Shot BLIP (model_base.pth - 129M pretraining, never fine-tuned on retrieval)."""
    return BLIP_BASELINE(device=device, ckpt=ckpt, mode='zs')


def BLIP_COCOFT(device="cuda", ckpt=None):
    """COCO Fine-tuned BLIP (model_base_retrieval_coco.pth)."""
    return BLIP_BASELINE(device=device, ckpt=ckpt, mode='cocoft')


def CLIP_SKETCH_ENCODER(device="cuda"):
    """Dedicated encoder for sketch branch — CLIP handles line drawings better than BLIP."""
    try:
        import open_clip
        model, _, preprocess = open_clip.create_model_and_transforms(
            'ViT-L-14', pretrained='openai')
        model = model.to(device).eval()

        def encode(image):
            with torch.no_grad():
                return F.normalize(model.encode_image(image), dim=-1)

        def prep(path):
            return preprocess(Image.open(path).convert('RGB'))

        return ImageEmbedder(encode, prep)
    except ImportError:
        try:
            import clip
            model, preprocess = clip.load("ViT-L/14", device=device)
            model.eval()

            def encode(image):
                with torch.no_grad():
                    return F.normalize(model.encode_image(image), dim=-1)

            def prep(path):
                return preprocess(Image.open(path).convert('RGB'))

            return ImageEmbedder(encode, prep)
        except ImportError:
            from transformers import CLIPVisionModelWithProjection, AutoProcessor
            clip_model_id = "openai/clip-vit-large-patch14"
            print(f"  [CLIP] Loading via transformers: {clip_model_id}")
            model = CLIPVisionModelWithProjection.from_pretrained(clip_model_id).to(device).eval()
            processor = AutoProcessor.from_pretrained(clip_model_id)

            def encode(image):
                with torch.no_grad():
                    outputs = model(pixel_values=image)
                    return F.normalize(outputs.image_embeds, dim=-1)

            def prep(path):
                raw = Image.open(path).convert('RGB')
                inputs = processor(images=raw, return_tensors="pt")
                return inputs['pixel_values'].squeeze(0)

            return ImageEmbedder(encode, prep)
