import torch
from basicsr.models import build_model
from basicsr.utils.options import parse

opt_path = "Motion_Deblurring/Options/Deblurring_Restormer.yml"

opt, _ = parse(opt_path, is_train=True)

opt["dist"] = False
opt["num_gpu"] = 1

model = build_model(opt)

checkpoint = torch.load(
    "Motion_Deblurring/pretrained_models/motion_deblurring.pth",
    map_location="cpu"
)

state_dict = checkpoint["params"]

model_state = model.net_g.state_dict()

missing = []
mismatched = []

for key, value in model_state.items():
    if key not in state_dict:
        missing.append(key)
    elif value.shape != state_dict[key].shape:
        mismatched.append(
            (key, tuple(value.shape), tuple(state_dict[key].shape))
        )

unexpected = [
    key for key in state_dict if key not in model_state
]

print("Missing keys:", len(missing))
print("Mismatched keys:", len(mismatched))
print("Unexpected keys:", len(unexpected))

if missing:
    print("Missing examples:", missing[:5])

if mismatched:
    print("Mismatch examples:", mismatched[:5])

if unexpected:
    print("Unexpected examples:", unexpected[:5])