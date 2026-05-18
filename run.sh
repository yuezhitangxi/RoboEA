gpu_id='1'
# 'FB15K_DB15K' 'FB15K_YAGO15K' 'zh_en' 'ja_en' 'fr_en'
dataset='FB15K_DB15K'
noise_name='Full_MHN'
noise_ratio=0.2
noise_ratio_tag=20
use_noisy_data=1
ratio=0.2
seed=2023
il_start=50
bsize=3500
adapter_choice=0
dropout=0.0
rank=8
use_GphForward=1
other_modal_type=0
Is_LMFSoftmax=0
joint_type=1
mr_fusion_type=7
final_fusion_type=(0)
if [[ "$dataset" == "FB15K_DB15K" ]]; then
    data_name='FBDB15K'
elif [[ "$dataset" == "FB15K_YAGO15K" ]]; then
    data_name='FBYG15K'
else
    data_name=${dataset}
fi
if [[ "$data_name" == *"FB"* ]]; then
    dataset_dir='mmkg'
    tau=0.1
    other_modal_type=0
else
    dataset_dir='DBP15K'
    tau=0.1
    ratio=0.3
fi
if [[ "${use_noisy_data}" == "1" ]]; then
    noisy_tag="${dataset}_${noise_name}_${noise_ratio_tag}"
    file_dir="data/noisy/${noisy_tag}"
    img_feature_path="data/mmkg/pkls/${noisy_tag}_GA_id_img_feature_dict.pkl"
else
    file_dir="data/${dataset_dir}/${data_name}/norm"
    img_feature_path=""
fi
echo "Running with dataset=${dataset}, ratio=${ratio}, file_dir=${file_dir}"
current_datetime=$(date +"%Y-%m-%d-%H-%M")
head_name=${current_datetime}_${dataset}
if [[ "${use_noisy_data}" == "1" ]]; then
    head_name=${head_name}_${noise_name}_${noise_ratio}
fi
file_name=${head_name}_${ratio}
echo ${file_name}
CUDA_VISIBLE_DEVICES=${gpu_id} python3 -u src/run.py \
    --file_dir ${file_dir} \
    ${img_feature_path:+--img_feature_path ${img_feature_path}} \
    --pred_name ${file_name} \
    --rate ${ratio} \
    --lr .0005 \
    --epochs 500 \
    --dropout ${dropout} \
    --hidden_units "300,300,300" \
    --check_point 50  \
    --bsize ${bsize} \
    --il_start ${il_start} \
    --semi_learn_step 5 \
    --csls \
    --csls_k 3 \
    --seed ${seed} \
    --tau ${tau} \
    --tau2 4.0 \
    --structure_encoder "Dualmodal-joint-LMF" \
    --joint_type ${joint_type} \
    --img_dim 300 \
    --attr_dim 300 \
    --name_dim 100 \
    --char_dim 100 \
    --bi_adapter \
    --adapter_choice ${adapter_choice} \
    --use_ms_loss \
    --use_joint_loss \
    --joint_use_nce \
    --use_cosface_loss \
    --cosface_margin 0.15 \
    --cosface_scale 2 \
    --cosface_hard_topk 10 \
    --cosface_hard_weight 0.05 \
    --cosface_hard_margin 0.15 \
    --cosface_focal_gamma 0 \
    --cosface_t_max 0.1 \
    --cosface_warmup_epoch 50 \
    --LMFrank ${rank} \
    --use_GphForward ${use_GphForward} \
    --add_other_modal ${other_modal_type} \
    --Is_LMFSoftmax ${Is_LMFSoftmax} \
    --mr_fusion_type ${mr_fusion_type} \
    --final_fusion_type ${final_fusion_type} \
    --use_proxy \
    --w_name \
    --w_char  > logs/${file_name}.log
echo $0 >> logs/${file_name}.log
