# Labeling Strategy for LoRA Fine-Tuning

What labels to use when training LoRA adapters for domain-specific knowledge.

---

## Key Principle: Labels Teach What You Want to Search

**Your labels define what the model learns.**

If you want to search "blue Tesla Model 3" → label training images as "blue Tesla Model 3"

You are NOT trying to recreate original training → You are teaching NEW domain knowledge

---

## Original EVA-02 Labels (Generic)

**EVA-02 was trained on web images with generic captions:**

```
Image: [Blue Tesla Model 3 car]
Original caption: "car parked on street"
                  "electric vehicle"
                  "my new ride"
                  "parked outside"
```

**Problem:** Labels don't specify brand, model, color precisely
→ Model learns general "car" concept, not specific brands

---

## Your Labels (Domain-Specific) ⭐

**For vehicle search, use specific attributes:**

```
Image: [Blue Tesla Model 3 car]
Your label: "blue Tesla Model 3 sedan"
            or "Tesla Model 3 in blue"
            or "blue Tesla electric sedan"
```

**Result:** Model learns:
- ✅ "Tesla" = specific brand
- ✅ "Model 3" = specific model
- ✅ "blue" = color attribute
- ✅ Association between visual appearance and brand name

---

## Label Format for CLIP Fine-Tuning

### CLIP trains on image-text pairs:

```python
training_data = [
    (image1, "blue Tesla Model 3 sedan"),
    (image2, "red Toyota Camry 2020"),
    (image3, "white BMW X5 SUV"),
    ...
]
```

**Text can be:**
- ✅ Short phrases: "Tesla Model 3"
- ✅ Descriptive sentences: "blue Tesla Model 3 sedan parked in lot"
- ✅ Natural language: "a blue electric car, Tesla Model 3"
- ✅ Template-based: "{color} {brand} {model} {type}"

**All work! Choose based on your needs.**

---

## Labeling Strategies

### Strategy 1: Simple Labels (Minimal)

**Format:** `{brand} {model}`

```
Training examples:
  - "Tesla Model 3"
  - "Toyota Camry"
  - "BMW X5"
  - "Honda Civic"
```

**Pros:**
- ✅ Easy to create
- ✅ Clean, consistent
- ✅ Teaches brand/model names

**Cons:**
- ⚠️ No color, year, type info
- ⚠️ Less expressive

**Use when:** You only care about make/model, not attributes

---

### Strategy 2: Attribute Labels (Recommended) ⭐

**Format:** `{color} {brand} {model} {type}`

```
Training examples:
  - "blue Tesla Model 3 sedan"
  - "red Toyota Camry sedan"
  - "white BMW X5 SUV"
  - "silver Honda Civic hatchback"
```

**Pros:**
- ✅ Rich information
- ✅ Supports attribute search ("blue cars", "SUVs")
- ✅ Better compositional understanding

**Cons:**
- ⚠️ More labeling effort

**Use when:** You want to search by attributes (color, type)

---

### Strategy 3: Natural Language (Maximum Context)

**Format:** Natural sentences

```
Training examples:
  - "a blue Tesla Model 3 electric sedan parked in parking lot"
  - "red Toyota Camry 2020 sedan on the street"
  - "white BMW X5 luxury SUV"
  - "person standing next to silver Honda Civic"
```

**Pros:**
- ✅ Most natural
- ✅ Adds scene context
- ✅ Better for complex queries

**Cons:**
- ⚠️ Harder to create consistently
- ⚠️ More verbose

**Use when:** You have detailed annotations or captions

---

### Strategy 4: Template-Based (Scalable)

**Use templates with variations:**

```python
templates = [
    "{color} {brand} {model}",
    "{brand} {model} in {color}",
    "{color} {brand} {model} {type}",
    "a {color} {type}, {brand} {model}",
]

# Generate labels
for img, metadata in dataset:
    template = random.choice(templates)
    label = template.format(
        color=metadata['color'],
        brand=metadata['brand'],
        model=metadata['model'],
        type=metadata['type']
    )
    
    training_data.append((img, label))

# Result:
#   "blue Tesla Model 3"
#   "Tesla Model 3 in blue"
#   "blue Tesla Model 3 sedan"
#   "a blue sedan, Tesla Model 3"
```

**Pros:**
- ✅ Automatic generation
- ✅ Natural variation (important!)
- ✅ Consistent attributes
- ✅ Scalable to thousands of images

**Cons:**
- ⚠️ Need structured metadata

**Use when:** You have metadata (brand, model, color) but not captions

---

## Example: Vehicle Dataset Labeling

### CompCars Dataset Structure:

```
compcars/
  data/
    image/
      1/              # Make ID (e.g., Acura)
        1/            # Model ID (e.g., Integra)
          2007/       # Year
            00001.jpg
            00002.jpg
```

### Label Generation Script:

```python
import csv
from pathlib import Path

# Mappings (from dataset)
makes = {1: "Acura", 2: "Aston Martin", 3: "Audi", ...}
models = {1: "Integra Type R", 2: "Integra", ...}

# Templates for variation
templates = [
    "{brand} {model}",
    "{brand} {model} {year}",
    "{brand} {model} {year} {type}",
    "a {brand} {model} car",
]

training_data = []

# Walk through dataset
for img_path in Path('compcars/data/image').rglob('*.jpg'):
    parts = img_path.parts
    make_id = int(parts[-4])
    model_id = int(parts[-3])
    year = parts[-2]
    
    brand = makes[make_id]
    model = models[model_id]
    
    # Generate label with random template
    template = random.choice(templates)
    label = template.format(
        brand=brand,
        model=model,
        year=year,
        type="car"  # or infer from model name
    )
    
    training_data.append((str(img_path), label))

# Save
with open('vehicle_training_data.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['image_path', 'caption'])
    writer.writerows(training_data)

# Output examples:
#   "Acura Integra Type R"
#   "Acura Integra Type R 2007"
#   "a Acura Integra Type R car"
#   "Aston Martin DB9 2006"
```

---

## Adding Color Information

**If dataset has color labels:**

```python
# From dataset or color detection
colors = {
    'img1.jpg': 'blue',
    'img2.jpg': 'red',
    'img3.jpg': 'white',
}

# Add to template
label = f"{colors[img]} {brand} {model}"
# "blue Tesla Model 3"
```

**If dataset doesn't have colors:**

**Option 1:** Use color detection model
```python
from colorthief import ColorThief

def get_dominant_color(img_path):
    ct = ColorThief(img_path)
    rgb = ct.get_color(quality=1)
    return rgb_to_name(rgb)  # e.g., (0,0,255) → "blue"

color = get_dominant_color(img_path)
label = f"{color} {brand} {model}"
```

**Option 2:** Manual annotation (for subset)
```python
# Label subset manually, train on those
important_cars = [
    ('car1.jpg', 'blue Tesla Model 3'),
    ('car2.jpg', 'red Tesla Model 3'),
    # ...
]
```

**Option 3:** Skip color initially
```python
# First pass: just brand/model
label = f"{brand} {model}"

# Later: add color as you collect more data
```

---

## Do You Need to Match Original EVA-02 Style? NO!

### Original EVA-02 Training:

```
Dataset: Merged-2B (2 billion web images)
Captions: Generic web captions
  - "car on the road"
  - "vehicle parked"
  - "my new car"
  - "photo of car"

Goal: Learn general visual-language alignment
```

### Your Fine-Tuning:

```
Dataset: 10K-100K vehicle images
Captions: Domain-specific labels
  - "blue Tesla Model 3 sedan"
  - "red Toyota Camry 2020"
  - "white BMW X5 SUV"

Goal: Teach specific brand/model knowledge
```

**You are teaching NEW knowledge, not mimicking old style!**

---

## Label Quality > Label Style

**What matters:**

✅ **Consistency:** Same naming for same concepts
```
Good:
  - "Tesla Model 3" (always)
  - "Tesla Model S" (always)
  
Bad:
  - "Tesla Model 3"
  - "model 3 tesla"  ← Inconsistent capitalization/order
  - "Tesla model3"   ← Inconsistent spacing
```

✅ **Specificity:** Include details you want to search
```
Good: "blue Tesla Model 3 sedan"
Bad:  "car"  ← Too generic
```

✅ **Accuracy:** Labels match images
```
Good: Image shows blue car → "blue Tesla Model 3"
Bad:  Image shows blue car → "red Tesla Model 3"  ← Wrong!
```

✅ **Variation:** Different phrasings of same concept
```
Good:
  - "blue Tesla Model 3"
  - "Tesla Model 3 in blue"
  - "a blue Tesla electric sedan"
  
Bad (too repetitive):
  - "blue Tesla Model 3"
  - "blue Tesla Model 3"
  - "blue Tesla Model 3"  ← Identical every time
```

---

## How Much Data Do You Need?

### Minimum (Quick Test):
- **1,000 images** with labels
- Good for: Proof of concept, test if approach works

### Recommended (Good Results):
- **10,000 images** with labels
- Good for: Production-quality adapter
- Coverage: 20-50 car brands, top models

### Optimal (Best Results):
- **50,000+ images** with labels
- Good for: Maximum accuracy
- Coverage: 100+ brands, hundreds of models

### Rule of thumb:
- **~100-500 examples per class** (brand/model)
- More common brands (Tesla, Toyota): more examples
- Rare brands (Bugatti, Koenigsegg): fewer examples OK

---

## Practical Labeling Workflow

### Automated (Best for Large Datasets):

```python
# 1. Extract metadata from dataset
metadata = parse_compcars_structure(dataset_path)

# 2. Generate labels with templates
labels = []
for img, meta in metadata:
    label = generate_label(
        brand=meta['brand'],
        model=meta['model'],
        year=meta['year'],
        template=random.choice(templates)
    )
    labels.append((img, label))

# 3. Save to CSV
save_training_csv(labels, 'vehicle_data.csv')
```

**Time:** Minutes (automated)

---

### Manual (Best for Small/Critical Datasets):

```python
# 1. Create labeling tool
from tkinter import Tk, Label, Entry, Button

def label_images(image_folder):
    labels = {}
    for img in image_folder.glob('*.jpg'):
        # Show image
        show_image(img)
        
        # User inputs label
        label = input(f"Label for {img.name}: ")
        labels[img] = label
    
    return labels

# 2. Label manually
labels = label_images('unlabeled_cars/')

# 3. Save
save_csv(labels, 'manual_labels.csv')
```

**Time:** ~1-5 seconds per image (10K images = 3-14 hours)

---

### Semi-Automated (Balanced):

```python
# 1. Auto-generate initial labels
auto_labels = generate_from_metadata(dataset)

# 2. Sample subset for manual review
sample = random.sample(auto_labels, k=500)

# 3. Manually verify/correct
for img, label in sample:
    show_image(img)
    print(f"Suggested: {label}")
    corrected = input("Correct label (or Enter to keep): ")
    if corrected:
        labels[img] = corrected

# 4. Use auto labels for rest
final_labels = auto_labels + corrected_labels
```

**Time:** Faster than full manual, more accurate than pure auto

---

## Example Training Data Files

### CSV Format (Recommended):

```csv
image_path,caption
/data/cars/tesla_model3_001.jpg,"blue Tesla Model 3 sedan"
/data/cars/toyota_camry_001.jpg,"red Toyota Camry 2020 sedan"
/data/cars/bmw_x5_001.jpg,"white BMW X5 SUV"
/data/cars/honda_civic_001.jpg,"silver Honda Civic hatchback"
```

### JSON Format (Alternative):

```json
[
  {
    "image": "/data/cars/tesla_model3_001.jpg",
    "caption": "blue Tesla Model 3 sedan",
    "metadata": {
      "brand": "Tesla",
      "model": "Model 3",
      "color": "blue",
      "type": "sedan"
    }
  },
  {
    "image": "/data/cars/toyota_camry_001.jpg",
    "caption": "red Toyota Camry 2020 sedan",
    "metadata": {
      "brand": "Toyota",
      "model": "Camry",
      "color": "red",
      "year": "2020",
      "type": "sedan"
    }
  }
]
```

---

## Testing Your Labels

### After labeling, validate quality:

```python
# 1. Load labels
with open('vehicle_data.csv') as f:
    labels = csv.DictReader(f)

# 2. Check consistency
brand_variations = defaultdict(set)
for row in labels:
    caption = row['caption']
    # Extract brand (simple regex)
    brand = extract_brand(caption)
    brand_variations[brand].add(caption.split()[0])

# Find inconsistencies
for brand, variations in brand_variations.items():
    if len(variations) > 1:
        print(f"Inconsistent: {brand} → {variations}")
        # e.g., "Tesla" vs "tesla" vs "TESLA"

# 3. Check coverage
brands = [extract_brand(row['caption']) for row in labels]
brand_counts = Counter(brands)
print("Top brands:", brand_counts.most_common(10))
print("Rare brands:", [b for b,c in brand_counts.items() if c < 10])

# 4. Visual spot check
random_samples = random.sample(labels, 20)
for row in random_samples:
    img = load_image(row['image_path'])
    print(f"Caption: {row['caption']}")
    show_image(img)
    input("Correct? (Enter to continue)")
```

---

## Summary

| Label Strategy | Format | Effort | Quality | Best For |
|----------------|--------|--------|---------|----------|
| **Simple** | "Tesla Model 3" | Low | Good | Brand/model search only |
| **Attribute** ⭐ | "blue Tesla Model 3 sedan" | Medium | Excellent | Full attribute search |
| **Natural Language** | "a blue electric sedan..." | High | Excellent | Complex queries |
| **Template-Based** | Auto-generated variations | Low | Very Good | Large datasets |

**Recommended:** **Attribute labels** with **template variation**

```python
templates = [
    "{color} {brand} {model}",
    "{brand} {model} in {color}",
    "{color} {type}, {brand} {model}",
]

# Generates natural variation:
#   "blue Tesla Model 3"
#   "Tesla Model 3 in blue"
#   "blue sedan, Tesla Model 3"
```

---

## Quick Start Script

```python
#!/usr/bin/env python3
"""
Generate vehicle training labels from CompCars dataset
"""

import csv
import random
from pathlib import Path

# Label templates
TEMPLATES = [
    "{brand} {model}",
    "{brand} {model} {year}",
    "{color} {brand} {model}",
    "{brand} {model} in {color}",
]

# Brand/model mappings (from CompCars metadata)
MAKES = {1: "Acura", 2: "Audi", 3: "BMW", 4: "Tesla", 5: "Toyota", ...}
MODELS = {1: "Integra", 2: "A4", 3: "X5", 4: "Model 3", 5: "Camry", ...}

def generate_labels(dataset_path, output_csv):
    """Generate training labels"""
    
    training_data = []
    
    # Walk through images
    for img_path in Path(dataset_path).rglob('*.jpg'):
        # Parse structure: .../make_id/model_id/year/image.jpg
        parts = img_path.parts
        make_id = int(parts[-4])
        model_id = int(parts[-3])
        year = parts[-2]
        
        # Get metadata
        brand = MAKES[make_id]
        model = MODELS[model_id]
        
        # Generate label with variation
        template = random.choice(TEMPLATES)
        label = template.format(
            brand=brand,
            model=model,
            year=year,
            color=""  # Add color detection if available
        ).strip()
        
        training_data.append((str(img_path), label))
    
    # Save
    with open(output_csv, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['image_path', 'caption'])
        writer.writerows(training_data)
    
    print(f"✓ Generated {len(training_data)} labels")
    print(f"✓ Saved to: {output_csv}")

if __name__ == '__main__':
    generate_labels(
        dataset_path='~/datasets/compcars/data/image',
        output_csv='vehicle_training_data.csv'
    )
```

---

**Last Updated:** 2026-05-13  
**Contact:** chinghokuk@gmail.com
