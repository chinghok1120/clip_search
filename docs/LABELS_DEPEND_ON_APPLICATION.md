# Labels Depend on Application & Template Usage

Understanding how labels should match your use case and how to use template variations.

---

## Question 1: Do Labels Depend on Application?

**YES! Labels define what the model learns to search for.**

### Example: The Damaged Tesla Image

**Image:** Crashed red Tesla Model 3 on highway

**Different applications → Different labels:**

#### Application 1: Vehicle Brand Identification
**Goal:** Identify car make/model regardless of condition

**Label:**
```
"red Tesla Model 3"
```

**Why:** Focus on brand/model, condition doesn't matter

**Search queries this enables:**
- "Tesla Model 3" → ✅ Finds it
- "red Tesla" → ✅ Finds it
- "damaged car" → ❌ Doesn't find it (not taught)

---

#### Application 2: Accident Detection
**Goal:** Find damaged/crashed vehicles

**Label:**
```
"damaged red Tesla Model 3"
"crashed Tesla Model 3"
"vehicle accident"
```

**Why:** Emphasize the damage/accident aspect

**Search queries this enables:**
- "damaged car" → ✅ Finds it
- "crashed Tesla" → ✅ Finds it
- "vehicle accident" → ✅ Finds it
- "Tesla Model 3" → ✅ Still finds it (brand included)

---

#### Application 3: Surveillance/Forensics
**Goal:** Search by scene, context, and objects

**Label:**
```
"crashed red Tesla Model 3 on highway"
"accident scene with damaged red Tesla"
```

**Why:** Include spatial and scene context

**Search queries this enables:**
- "accident on highway" → ✅ Finds it
- "damaged car on road" → ✅ Finds it
- "Tesla crash" → ✅ Finds it

---

### Rule: Labels Define Search Capabilities

```
What you label = What you can search

Label: "blue Tesla Model 3"
→ Can search: "blue car", "Tesla", "Model 3"
→ Can't search: "car in parking lot" (location not labeled)

Label: "blue Tesla Model 3 in parking lot"
→ Can search: "blue car", "Tesla", "Model 3", "car in parking lot"
→ Can't search: "damaged car" (condition not labeled)

Label: "damaged blue Tesla Model 3 in parking lot"
→ Can search: Everything above + "damaged car", "crashed vehicle"
```

**Your labels should match your intended queries!**

---

## Designing Labels for Your Application

### Step 1: Define What You Want to Search

**Ask yourself:**
- What queries will users make?
- What information do I need to retrieve?
- What's most important: brand? condition? location? people?

### Step 2: Label Accordingly

**Example scenarios:**

#### Scenario A: Car Dealership Inventory
**Queries:** "blue sedan", "Tesla Model 3", "SUV in stock"

**Labels:**
```
"{color} {brand} {model} {type}"
"blue Tesla Model 3 sedan"
"red Toyota Camry sedan"
"white BMW X5 SUV"
```

**Skip:** Location (cars all in same lot), damage (inventory is clean)

---

#### Scenario B: Parking Lot Surveillance
**Queries:** "car at entrance", "vehicle in lot A", "person near car"

**Labels:**
```
"person next to {color} {brand} {model} in {location}"
"person next to blue Tesla Model 3 in lot A"
"red Toyota Camry in lot B"
"white BMW X5 at entrance"
```

**Skip:** Detailed brand/model (if not important), focus on location/people

---

#### Scenario C: Insurance Claim Processing
**Queries:** "damaged car", "front-end collision", "vehicle accident"

**Labels:**
```
"{damage_type} {color} {brand} {model}"
"front-end damage red Tesla Model 3"
"rear collision white BMW X5"
"side damage blue Toyota Camry"
```

**Skip:** Location (less relevant), focus on damage

---

#### Scenario D: Your Surveillance System
**Queries:** Mix of vehicle ID + location + people

**Labels:**
```
"{color} {brand} {model}"                    # Vehicle ID
"person next to {color} {brand}"             # People + vehicle
"{brand} at {location}"                      # Location
"{color} {brand} {model} in {location}"      # Full context
```

**Include:** Brand/model (for ID), location (surveillance zones), people (interactions)

---

## Question 2: Template Usage - All Combinations or Just One?

### SHORT ANSWER:

**❌ DON'T:** Use all templates for the same image  
**✅ DO:** Use different templates across different images (natural variation)

---

### The Goal: Natural Language Variation

**You want the model to learn:**
- "blue Tesla Model 3" = "Tesla Model 3 in blue" = "blue Tesla sedan"
- Different phrasings → Same meaning
- Robust to different query styles

**How to achieve this:** Vary templates across dataset

---

### Strategy 1: Random Template per Image (Recommended) ⭐

**Approach:** Pick ONE random template per image

```python
import random

templates = [
    "{color} {brand} {model}",
    "{brand} {model} in {color}",
    "{color} {brand} {model} {type}",
    "person next to {color} {brand} {model}",
]

# For each image, pick ONE template
for img, metadata in dataset:
    template = random.choice(templates)  # Random selection
    
    label = template.format(
        color=metadata['color'],
        brand=metadata['brand'],
        model=metadata['model'],
        type=metadata['type']
    )
    
    training_data.append((img, label))

# Result:
# img1.jpg, "blue Tesla Model 3"              ← template 1
# img2.jpg, "Toyota Camry in red"             ← template 2
# img3.jpg, "white BMW X5 SUV"                ← template 3
# img4.jpg, "person next to silver Honda Civic" ← template 4
# img5.jpg, "black Audi A4"                   ← template 1 (random again)
```

**Why this works:**
- ✅ Each image has ONE label (no duplication)
- ✅ Across dataset, natural variation exists
- ✅ Model learns different phrasings
- ✅ No wasted compute (no duplicate images)

---

### Strategy 2: Context-Aware Template Selection

**Approach:** Choose template based on image content

```python
def select_template(img, metadata):
    """Smart template selection based on image content"""
    
    # Detect image content
    has_person = detect_person(img)
    location = detect_location(img)
    
    # Select appropriate template
    if has_person:
        return "person next to {color} {brand} {model}"
    elif location == "parking_lot":
        return "{color} {brand} {model} in parking lot"
    elif location == "gas_station":
        return "{color} {brand} {model} at gas station"
    else:
        return "{color} {brand} {model}"

# Apply
for img, metadata in dataset:
    template = select_template(img, metadata)
    label = template.format(**metadata)
    training_data.append((img, label))

# Result:
# img1.jpg (has person), "person next to blue Tesla Model 3"
# img2.jpg (parking lot), "red Toyota Camry in parking lot"
# img3.jpg (plain),      "white BMW X5"
```

**Why this works:**
- ✅ Templates match image content (accurate)
- ✅ Natural variation
- ✅ More meaningful labels

---

### Strategy 3: Template Distribution (Advanced)

**Approach:** Control distribution of template types

```python
# Define template weights
template_weights = [
    ("{color} {brand} {model}", 0.4),              # 40% simple
    ("{color} {brand} {model} {type}", 0.3),       # 30% with type
    ("person next to {color} {brand}", 0.2),       # 20% with person
    ("{color} {brand} at {location}", 0.1),        # 10% with location
]

# Sample with weights
for img, metadata in dataset:
    template = random.choices(
        [t[0] for t in template_weights],
        weights=[t[1] for t in template_weights]
    )[0]
    
    label = template.format(**metadata)
    training_data.append((img, label))

# Result: Controlled distribution
# - 40% will be "blue Tesla Model 3"
# - 30% will be "blue Tesla Model 3 sedan"
# - 20% will be "person next to blue Tesla"
# - 10% will be "blue Tesla at parking lot"
```

---

### ❌ WRONG Approach: All Templates for Same Image

**DON'T DO THIS:**

```python
# BAD: Creating multiple copies of same image
templates = [
    "{color} {brand} {model}",
    "person next to {color} {brand}",
    "{color} {brand} in parking lot",
    "{color} {brand} at gas station",
]

# For EACH image, use ALL templates (WRONG!)
for img, metadata in dataset:
    for template in templates:  # ← BAD! Creates 4 copies
        label = template.format(**metadata)
        training_data.append((img, label))

# Result:
# img1.jpg, "blue Tesla Model 3"                  ← same image
# img1.jpg, "person next to blue Tesla"           ← same image
# img1.jpg, "blue Tesla in parking lot"           ← same image
# img1.jpg, "blue Tesla at gas station"           ← same image
```

**Why this is bad:**
- ❌ 4× more training data (slower training)
- ❌ Overfitting to specific images
- ❌ Wastes compute
- ❌ Some labels might be wrong (no person in image, but labeled "person next to")

---

### Exception: Data Augmentation with Multiple Labels

**When it IS OK to use multiple labels:**

If labels describe **different valid aspects** of the same image:

```python
# Image shows: Red Tesla Model 3 with person standing next to it

# Multiple valid labels (all true):
labels = [
    "red Tesla Model 3",                    # ✓ Valid
    "person next to red Tesla",             # ✓ Valid
    "Tesla Model 3 in parking lot",         # ✓ Valid (if in lot)
]

# Add all (they're all accurate descriptions)
for label in labels:
    training_data.append((img, label))
```

**Use this when:**
- ✅ All labels are **factually accurate**
- ✅ You want to emphasize multiple aspects
- ✅ Labels describe **different valid searches** for same image

**But typically:** Pick ONE template per image (random) is cleaner

---

## Practical Examples

### Example 1: Your 2 Tesla Images

**Image 1:** `tesla-model3.jpg` (red Tesla on highway)

**Application A - Vehicle ID:**
```python
label = "red Tesla on highway"  # One label
```

**Application B - Traffic monitoring:**
```python
label = "red Tesla in traffic"  # One label
```

**Application C - Both (multiple valid):**
```python
labels = [
    "red Tesla on highway",
    "red Tesla in traffic",
]
# Both valid, use both if you want
```

---

**Image 2:** `tesla-model3-ragged.jpg` (damaged Tesla)

**Application A - Vehicle ID only:**
```python
label = "red Tesla Model 3"  # Ignore damage
```

**Application B - Accident detection:**
```python
label = "damaged red Tesla Model 3"  # Emphasize damage
```

**Application C - Forensics (multiple aspects):**
```python
labels = [
    "damaged red Tesla Model 3",
    "crashed vehicle on highway",
    "accident scene red Tesla",
]
# Multiple valid labels for different search needs
```

---

## Template Usage Summary

### ✅ DO:

**One random template per image:**
```python
for img in dataset:
    template = random.choice(templates)
    label = apply_template(template, img_metadata)
    add_to_training(img, label)
```

**Or context-aware selection:**
```python
for img in dataset:
    template = smart_select(img)  # Based on image content
    label = apply_template(template, img_metadata)
    add_to_training(img, label)
```

**Or multiple labels if all valid:**
```python
for img in dataset:
    valid_labels = generate_all_valid_labels(img)
    for label in valid_labels:
        add_to_training(img, label)
```

---

### ❌ DON'T:

**All templates blindly:**
```python
for img in dataset:
    for template in templates:  # ← BAD!
        label = apply_template(template, img_metadata)
        add_to_training(img, label)
# Creates invalid labels (e.g., "person next to" when no person)
```

---

## Decision Flow

```
┌─────────────────────────────────────┐
│ What do you want to search for?     │
└───────────┬─────────────────────────┘
            ↓
┌─────────────────────────────────────┐
│ Define label format to match        │
│ Example: "{color} {brand} {model}"  │
└───────────┬─────────────────────────┘
            ↓
┌─────────────────────────────────────┐
│ For each image:                     │
│ - Pick ONE template (random or      │
│   smart selection)                  │
│ - Or use multiple if all valid      │
└───────────┬─────────────────────────┘
            ↓
┌─────────────────────────────────────┐
│ Result: Dataset with natural        │
│ variation in label phrasing         │
└─────────────────────────────────────┘
```

---

## Recommendations

### For Your Surveillance System:

**1. Define your main queries:**
```
- "blue Tesla Model 3" (vehicle ID)
- "person next to car" (interaction)
- "car in parking lot A" (location)
- "damaged vehicle" (condition)
```

**2. Create matching templates:**
```python
templates = [
    "{color} {brand} {model}",                      # Vehicle ID
    "person next to {color} {brand}",               # Interaction
    "{color} {brand} in {location}",                # Location
    "damaged {color} {brand}",                      # Condition
]
```

**3. Use random or context-aware selection:**
```python
for img, meta in dataset:
    # Smart selection
    if meta['has_person']:
        template = "person next to {color} {brand}"
    elif meta['damaged']:
        template = "damaged {color} {brand}"
    elif meta['location']:
        template = "{color} {brand} in {location}"
    else:
        template = "{color} {brand} {model}"
    
    label = template.format(**meta)
    training_data.append((img, label))
```

**4. Result: One label per image, natural variation across dataset** ✅

---

**Last Updated:** 2026-05-13  
**Contact:** chinghokuk@gmail.com
