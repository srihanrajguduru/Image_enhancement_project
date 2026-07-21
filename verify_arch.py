import torch
from src.core.registry import registry

print("Testing DehazeFormer (Dehazing)...")
dehaze_model = registry.get_model("haze")
dehaze_model.load()
dummy_img = torch.randn(1, 3, 256, 256)
out_haze = dehaze_model.restore(dummy_img)
print(f"Dehaze output shape: {out_haze.shape}, Min: {out_haze.min().item():.3f}, Max: {out_haze.max().item():.3f}")

print("\nTesting Restormer (Deraining)...")
derain_model = registry.get_model("rain")
derain_model.load()
out_rain = derain_model.restore(dummy_img)
print(f"Derain output shape: {out_rain.shape}, Min: {out_rain.min().item():.3f}, Max: {out_rain.max().item():.3f}")

print("\nArchitecture upgrade successful!")
