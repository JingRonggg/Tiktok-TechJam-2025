from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
import torch
import torchvision.transforms as T
from PIL import Image
import io
import foolbox as fb
import numpy as np

app = FastAPI()


def get_model():
    model = torch.hub.load("pytorch/vision", "resnet18", pretrained=True)
    model.eval()
    return model


model = get_model()
fmodel = fb.PyTorchModel(model, bounds=(0, 1))
attack = fb.attacks.FGSM()

transform = T.Compose(
    [
        T.Resize((224, 224)),
        T.ToTensor(),
    ]
)


def image_to_tensor(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = transform(image)
    return tensor.unsqueeze(0)


@app.post("/attack")
async def adversarial_attack(file: UploadFile = File(...)):
    image_bytes = await file.read()
    x = image_to_tensor(image_bytes)
    x = x.clone().detach()
    label = model(x).argmax(dim=1)
    raw, clipped, is_adv = attack(fmodel, x, label, epsilons=0.1)
    adv_img = clipped.squeeze().detach().cpu().numpy()
    adv_img = np.transpose(adv_img, (1, 2, 0))
    adv_img = (adv_img * 255).astype(np.uint8)
    adv_pil = Image.fromarray(adv_img)
    buf = io.BytesIO()
    adv_pil.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")


@app.get("/")
def root():
    return {"message": "Adversarial Attack API (FGSM) is running."}
