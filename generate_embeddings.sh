echo "generate GSE embeddings."
python generate_embeddings.py --csv path/to/your/csv/with/sticker_id/and/sticker_id_2/columns \
    --image_folder /path/to/StickerQueries/image/folder \
    --output "path/to/output/embeddings" \
    --model_path 'openai-clip-vit-large-patch14' \
    --checkpoint_path "./weights/img_txt_combined_best.pt"
