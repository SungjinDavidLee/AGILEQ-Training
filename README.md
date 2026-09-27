# AGILEQ-Training

[한국어](README_ko.md)

CARLA data collection and model training for autonomous driving.

| Directory | Purpose | Status |
| --- | --- | --- |
| [data_gen](data_gen/README.md) | Collect camera images, BEV labels, and driving metadata in CARLA. | Included |
| [bevformer](bevformer/README.md) | Train multi-camera BEV perception models. | Source included; data2.py required |
| [TLCformer](TLCformer/README.md) | Train a temporal front-camera model for traffic-light GO/STOP state and distance bins. | Included |
| `RL_Training/` | Train autonomous-driving policies with reinforcement learning. | Planned |

Each module has its own setup and data requirements. Run its commands from that module's directory. See the module READMEs before connecting generated data to training: camera names, labels, and spatial conventions must agree.

English is the default documentation language; Korean documentation uses `README_ko.md`. Python comments and docstrings have been removed. Third-party Python header notices are preserved in the corresponding `THIRD_PARTY_NOTICES.txt` files. Generated Python caches and platform-specific BEVFormer build artifacts are excluded; extension sources are included.
