from pathlib import Path

root = Path("/Volumes/T9/iowa_research/Han_AIR_Dec_2025/data_volumes/data_all_volumes2")
expected = 1024*1536*512*2
plane = 1536*512*2

for p in sorted(root.glob("*.img")):
    sz = p.stat().st_size
    if sz != expected:
        print(
            p.name,
            "size=", sz,
            "delta=", sz-expected,
            "z_equiv=", sz/plane,
            "plane_remainder=", sz % plane,
        )