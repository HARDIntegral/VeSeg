import veseg

result = veseg.predict(
    "test_images/structure1.png",
    mode=veseg.RED,
)

print(result.mask.shape)

scaled = result.scaled(2)
print(scaled.mask.shape)

clean = result.despeckle()
print(clean.mask.shape)

clean.save_mask("test.png")