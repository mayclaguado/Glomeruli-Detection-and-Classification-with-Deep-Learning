# Glomeruli-Detection-and-Classification-with-Deep-Learning
Turns WSIs into glomeruli findings: downsampling, tissue masking, and overlapping tiling. YOLOv8 detects, we map results to slide coordinates, flag complete vs edge boxes, and deduplicate for a reliable count. All this for a 4-class LN classifier. Finally, it exports color overlays and audit-ready tables.
