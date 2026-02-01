batch_size = 1
modelname = "groundingdino"
backbone = "swin_T_224_1k"
position_embedding = "sine"
pe_temperatureH = 20
pe_temperatureW = 20
return_interm_indices = [1, 2, 3]
backbone_freeze_keywords = None
enc_layers = 6
dec_layers = 6
pre_norm = False
dim_feedforward = 2048
hidden_dim = 256
dropout = 0.0
nheads = 8
num_queries = 900
query_dim = 4
num_patterns = 0
num_feature_levels = 4
enc_n_points = 4
dec_n_points = 4
two_stage_type = "standard"
two_stage_bbox_embed_share = False
two_stage_class_embed_share = False
transformer_activation = "relu"
dec_pred_bbox_embed_share = True
dn_box_noise_scale = 1.0
dn_label_noise_ratio = 0.5
dn_label_coef = 1.0
dn_bbox_coef = 1.0
embed_init_tgt = True
dn_labelbook_size = 2000
max_text_len = 256
text_encoder_type = "weights/bert-base-uncased"
use_text_enhancer = True
use_fusion_layer = True
use_checkpoint = True
use_transformer_ckpt = True
use_text_cross_attention = True
text_dropout = 0.0
fusion_dropout = 0.0
fusion_droppath = 0.1
sub_sentence_present = True

# =========================
# Added for inference (minimal complete config)
# =========================

# basic
aux_loss = False
use_dn = False

# ---------------- added for inference compatibility ----------------
# matcher / loss related (required by models/GroundingDINO/matcher.py)
matcher_type = "HungarianMatcher"
set_cost_class = 2.0
set_cost_bbox = 5.0
set_cost_giou = 2.0
focal_alpha = 0.25

# ---------------- added for inference compatibility (loss weights) ----------------
cls_loss_coef = 1.0
bbox_loss_coef = 5.0
giou_loss_coef = 2.0

# for two-stage / denoising or intermediate losses (safe defaults)
interm_loss_coef = 1.0
no_interm_box_loss = False

# ---------------- added for focal loss ----------------
focal_alpha = 0.25
focal_gamma = 2.0

# postprocess / inference
num_select = 300          # 每张图最多保留多少个框（OGC 官方常用 300）
nms_iou_threshold = 0.5   # NMS IoU 阈值

# denoising (inference 用不到，但模型会访问)
dn_labelbook_size = 100
dn_number = 0
dn_box_noise_scale = 0.0
dn_label_noise_ratio = 0.0

# loss weights（inference 不用，但 build 时会读）
cls_loss_coef = 1.0
bbox_loss_coef = 5.0
giou_loss_coef = 2.0

# transformer / decoder
dec_layers = 6
num_queries = 900  # 必须和 checkpoint 一致（你这个权重是 900）

# eval flags (needed by PostProcess)
use_coco_eval = False
use_odinw_eval = False
use_lvis_eval = False

# for PostProcess compatibility (inference doesn't need fixed categories)
label_list = []