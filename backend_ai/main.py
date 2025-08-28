from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
import torch
import torchvision.transforms as T
from PIL import Image
import io
import foolbox as fb
import numpy as np
from transformers import CLIPProcessor, CLIPModel

app = FastAPI()


def get_model():
    model = torch.hub.load("pytorch/vision", "resnet18", pretrained=True)
    model.eval()
    return model


def get_streetclip_model():
    model = CLIPModel.from_pretrained("geolocal/StreetCLIP")
    processor = CLIPProcessor.from_pretrained("geolocal/StreetCLIP")
    return model, processor


model = get_model()
streetclip_model, streetclip_processor = get_streetclip_model()
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


@app.post("/results")
async def analysis_result(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    choices = [
        "Albania",
        "Iceland",
        "Puerto Rico",
        "Andorra",
        "India",
        "Romania",
        "Argentina",
        "Indonesia",
        "Russia",
        "Australia",
        "Ireland",
        "Rwanda",
        "Austria",
        "Israel",
        "Senegal",
        "Bangladesh",
        "Italy",
        "Serbia",
        "Belgium",
        "Japan",
        "Singapore",
        "Bermuda",
        "Jordan",
        "Slovakia",
        "Bhutan",
        "Kenya",
        "Slovenia",
        "Bolivia",
        "Kyrgyzstan",
        "South Africa",
        "Botswana",
        "Laos",
        "South Korea",
        "Brazil",
        "Latvia",
        "Spain",
        "Bulgaria",
        "Lesotho",
        "Sri Lanka",
        "Cambodia",
        "Lithuania",
        "Swaziland",
        "Canada",
        "Luxembourg",
        "Sweden",
        "Chile",
        "Macedonia",
        "Switzerland",
        "China",
        "Madagascar",
        "Taiwan",
        "Colombia",
        "Malaysia",
        "Thailand",
        "Croatia",
        "Malta",
        "Tunisia",
        "Czech Republic",
        "Mexico",
        "Turkey",
        "Denmark",
        "Monaco",
        "Uganda",
        "Dominican Republic",
        "Mongolia",
        "Ukraine",
        "Ecuador",
        "Montenegro",
        "United Arab Emirates",
        "Estonia",
        "Netherlands",
        "United Kingdom",
        "Finland",
        "New Zealand",
        "United States",
        "France",
        "Nigeria",
        "Uruguay",
        "Germany",
        "Norway",
        "Ghana",
        "Pakistan",
        "Greece",
        "Palestine",
        "Greenland",
        "Peru",
        "Guam",
        "Philippines",
        "Guatemala",
        "Poland",
        "Hungary",
        "Portugal",
    ]

    inputs = streetclip_processor(
        text=choices, images=image, return_tensors="pt", padding=True
    )

    outputs = streetclip_model(**inputs)
    logits_per_image = outputs.logits_per_image  # image-text similarity score
    probs = logits_per_image.softmax(dim=1)  # get label probabilities

    pred_idx = torch.argmax(probs, dim=1).item()
    top_prediction = choices[pred_idx]
    top_confidence = float(probs[0, pred_idx])

    all_predictions = []
    for name, p in zip(choices, probs[0].tolist()):
        rounded_confidence = round(p, 3)
        if rounded_confidence > 0:
            all_predictions.append({"location": name, "confidence": rounded_confidence})

    all_predictions.sort(key=lambda x: x["confidence"], reverse=True)

    return {
        "top_prediction": {
            "location": top_prediction,
            "confidence": round(top_confidence, 3),
        },
        "all_predictions": all_predictions,
    }


@app.get("/")
def root():
    return {"message": "Adversarial Attack API (FGSM) is running."}
