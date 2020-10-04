TOTAL_NUM_UPDATES=40000
WARMUP_UPDATES=500
LR=3e-5
MAX_TOKENS=64
UPDATE_FREQ=4
BART_PATH=bart.large/model.pt
#BART_PATH=checkpoints/checkpoint_best.pt

CUDA_VISIBLE_DEVICES=1 fairseq-train bart/plain-bin \
    --disc-epochs 10 --max-epoch 20 --recon-weight 0.0 --cycle-weight 0.6 \
    --save-dir checkpoints --weight-forward 0.8 --best-checkpoint-metric nll_loss \
    --restore-file $BART_PATH \
    --max-tokens $MAX_TOKENS \
    --task translation \
    --source-lang source --target-lang target \
    --truncate-source \
    --layernorm-embedding \
    --share-all-embeddings \
    --share-decoder-input-output-embed \
    --reset-optimizer --reset-dataloader --reset-meters \
    --required-batch-size-multiple 1 \
    --arch bart_large \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 \
    --dropout 0.1 --attention-dropout 0.1 \
    --weight-decay 0.01 --patience 3  --optimizer adam --adam-betas "(0.9, 0.999)" --adam-eps 1e-08 \
    --clip-norm 0.1 \
    --no-epoch-checkpoints --keep-best-checkpoints 3 --keep-last-epochs 0 --no-last-checkpoints \
    --lr-scheduler polynomial_decay --lr $LR --total-num-update $TOTAL_NUM_UPDATES --warmup-updates $WARMUP_UPDATES \
    --update-freq $UPDATE_FREQ \
    --skip-invalid-size-inputs-valid-test \
    --find-unused-parameters;
