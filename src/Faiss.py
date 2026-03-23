import faiss

dimension = embeddings.shape[1]  # 384

index = faiss.IndexFlatIP(dimension)
index.add(embeddings)

print(f"Index type: {type(index)}")
print(f"Total vectors indexed: {index.ntotal}")

##############

faiss.write_index(index, "faiss_index.bin")
print(f"Saved faiss_index.bin")
print(f"File size: {os.path.getsize('faiss_index.bin') / 1024 / 1024:.1f} MB")

##############

query = "moisturizer for dry winter skin"

query_embedding = model.encode(
    [query],
    normalize_embeddings=True,
    convert_to_numpy=True
)

scores, indices = index.search(query_embedding, k=5)

print(f"Query: '{query}'\n")
for rank, (idx, score) in enumerate(zip(indices[0], scores[0])):
    title = df.iloc[idx]["title"]
    print(f"{rank+1}. Score: {score:.4f} | {title}")
    
##############
#Output for quick Query Check:
'''''
Query: 'moisturizer for dry winter skin'

1. Score: 0.6637 | Body Prescriptions Intense Moisture Hand Cream/Moisturizer for Men for Rough and Dry Skin… (Intense Moisture)
2. Score: 0.6615 | Deep Moisture Body Oil for Skin & Hair, Winter Formula. (2 oz.) The luxuriously soothing all natural body oil for dry sensitive skin.
3. Score: 0.6350 | Night Cream Multi-Performance snow white face creme 50g Anti-drying facial moisturizer Daily Use Sun protection Skin tender Eliminate horny fast absorbing
4. Score: 0.6321 | Physicians Formula Elastin & Collagen Moisture Lotion, 4 oz
5. Score: 0.6310 | Mederma Advanced Dry Skin Therapy Hand & Body Lotion - 6 oz, Pack of 2''''
