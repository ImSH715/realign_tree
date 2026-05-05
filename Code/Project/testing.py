from src.data.patches import RandomPatchTifDataset

ds = RandomPatchTifDataset(
    root_dir="/mnt/parscratch/users/acb20si/2025_Forge/OSINFOR_data/01. Ortomosaicos/2023",
    patches_per_image=1,
    tile_cache_size=0
)

views, meta = ds[0]

print(len(views))        # should be num_views
print(views[0].shape)   # should be tensor
print(meta)
