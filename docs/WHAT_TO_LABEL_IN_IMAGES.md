# What to Label in Images: Focus vs Comprehensive

When fine-tuning with LoRA, should you label everything in the image or just your domain?

---

## Key Principle: You're Teaching NEW Knowledge

**Base CLIP already knows:**
- ✅ "person", "people", "man", "woman"
- ✅ "parking lot", "street", "building"
- ✅ "tree", "sky", "road"
- ✅ General objects and scenes

**Your vehicle LoRA is teaching:**
- 🆕 "Tesla" = specific car brand
- 🆕 "Model 3" = specific model
- 🆕 "Toyota Camry" = specific vehicle
- 🆕 Visual features of each brand

**Conclusion: Focus labels on what's NEW (vehicles), include context if relevant**

---

## Labeling Strategies

### Strategy 1: Vehicle-Only Labels (Minimal)

**Approach:** Label only the vehicle, ignore everything else

```
Image: [Blue Tesla Model 3 with person standing next to it]
Label: "blue Tesla Model 3"

Image: [Red Toyota Camry in parking lot with buildings]
Label: "red Toyota Camry"

Image: [White BMW X5, people in background]
Label: "white BMW X5"
```

**Pros:**
- ✅ Simple, fast to generate
- ✅ Focused on teaching vehicle knowledge
- ✅ Easiest to automate

**Cons:**
- ⚠️ Loses spatial context
- ⚠️ "person next to tesla" queries might not work

**Use when:**
- You only care about vehicle identification
- Quick labeling needed
- Vehicle is always the main subject

---

### Strategy 2: Vehicle + Important Context (Recommended) ⭐

**Approach:** Label vehicle + significant spatial/contextual elements

```
Image: [Blue Tesla Model 3 with person standing next to it]
Label: "person standing next to blue Tesla Model 3"

Image: [Red Toyota Camry in parking lot]
Label: "red Toyota Camry in parking lot"

Image: [White BMW X5 at gas station]
Label: "white BMW X5 at gas station"

Image: [Tesla being delivered, delivery person nearby]
Label: "delivery person with blue Tesla Model 3"
```

**What to include:**
- ✅ Primary object: Vehicle (with attributes)
- ✅ Important spatial relationships: "person next to", "car at", "parked near"
- ✅ Significant context: "parking lot", "gas station", "driveway"
- ❌ Skip: Minor background details

**Pros:**
- ✅ Supports compositional queries
- ✅ Maintains spatial understanding
- ✅ Still focused on vehicles
- ✅ Reasonable labeling effort

**Cons:**
- ⚠️ Slightly more complex labels

**Use when:** You want flexible search with context (recommended for most cases)

---

### Strategy 3: Comprehensive Labels (Maximum Detail)

**Approach:** Label everything in the image

```
Image: [Blue Tesla Model 3 with person in red jacket]
Label: "person in red jacket standing next to blue Tesla Model 3 sedan parked in parking lot with trees in background and cloudy sky"
```

**Pros:**
- ✅ Maximum information
- ✅ Rich compositional understanding

**Cons:**
- ❌ Very time-consuming
- ❌ Overkill for vehicle adapter
- ❌ Teaches base knowledge (which CLIP already knows!)

**Use when:** You need extremely detailed scene understanding (rare)

---

## What Base CLIP Already Knows

**Remember: You're fine-tuning, not training from scratch!**

Base EVA-02-B already learned from 2 billion images:

### General Objects:
```
✅ person, people, man, woman, child
✅ building, house, tree, road, sky
✅ parking lot, street, sidewalk
✅ jacket, clothing, accessories
```

### Spatial Relationships:
```
✅ "person standing next to"
✅ "car parked in"
✅ "vehicle at"
✅ "near", "beside", "behind"
```

### Scenes:
```
✅ parking lot, street, highway
✅ gas station, driveway
✅ indoor, outdoor
```

**You DON'T need to re-teach these!**

---

## What You're Teaching (NEW Knowledge)

### Vehicle Brands (Base CLIP is weak here):
```
🆕 Tesla, Toyota, BMW, Honda, Ford
🆕 Model 3, Camry, X5, Civic
🆕 Visual features: Tesla grille, BMW kidney grille
```

### Vehicle-Specific Attributes:
```
🆕 sedan, SUV, truck, hatchback
🆕 Brand-specific colors
🆕 Model years, generations
```

**Focus labels on teaching THIS!**

---

## Practical Examples

### Example 1: Person Next to Car

**Image:** Person in red jacket standing next to blue Tesla Model 3

**Option A (vehicle-only):**
```
"blue Tesla Model 3"
```
- ✅ Teaches: Tesla brand, blue color
- ❌ Loses: Person context

**Option B (with context) ⭐:**
```
"person next to blue Tesla Model 3"
or
"blue Tesla Model 3 with person standing beside it"
```
- ✅ Teaches: Tesla brand, blue color
- ✅ Keeps: Spatial relationship
- ✅ Base CLIP handles "person" part

**Option C (over-detailed):**
```
"person in red jacket and blue jeans standing next to blue Tesla Model 3 sedan in parking lot"
```
- ⚠️ Overkill (base CLIP knows "red jacket", "blue jeans", "parking lot")

**Recommendation:** Option B

---

### Example 2: Car in Parking Lot

**Image:** White BMW X5 SUV parked in crowded parking lot

**Option A (vehicle-only):**
```
"white BMW X5"
```

**Option B (with context) ⭐:**
```
"white BMW X5 in parking lot"
or
"white BMW X5 SUV parked in lot"
```

**Option C (over-detailed):**
```
"white BMW X5 luxury SUV parked in crowded parking lot with other vehicles and buildings in background"
```

**Recommendation:** Option B

---

### Example 3: Delivery Scene

**Image:** Delivery person in uniform handing keys next to red Toyota Camry

**Option A (vehicle-only):**
```
"red Toyota Camry"
```
- Loses important scene context

**Option B (with context) ⭐:**
```
"delivery person with red Toyota Camry"
or
"red Toyota Camry being delivered"
```
- Captures the scene while focusing on vehicle

**Option C (over-detailed):**
```
"delivery person in brown uniform handing keys to customer next to red Toyota Camry 2020 sedan in residential driveway"
```

**Recommendation:** Option B

---

## Label Templates with Context

### Basic Template (vehicle-only):
```python
templates = [
    "{color} {brand} {model}",
    "{brand} {model} {type}",
]

# Examples:
# "blue Tesla Model 3"
# "Toyota Camry sedan"
```

### Context Template (recommended) ⭐:
```python
templates = [
    "{color} {brand} {model}",
    "{color} {brand} {model} in {location}",
    "person next to {color} {brand} {model}",
    "{color} {brand} {model} at {location}",
    "{brand} {model} parked in {location}",
]

# Examples:
# "blue Tesla Model 3"
# "blue Tesla Model 3 in parking lot"
# "person next to blue Tesla Model 3"
# "blue Tesla Model 3 at gas station"
# "Tesla Model 3 parked in driveway"
```

### Advanced Template (optional):
```python
templates = [
    "{actor} {action} {color} {brand} {model}",
    "{color} {brand} {model} {state} in {location}",
]

# Examples:
# "person standing next to blue Tesla Model 3"
# "blue Tesla Model 3 parked in parking lot"
# "delivery person with red Toyota Camry"
```

---

## When to Include Context

### Include spatial relationships if:
- ✅ Relevant to surveillance queries ("person near car")
- ✅ Helps distinguish scenes ("car at gas station" vs "car in driveway")
- ✅ Important for your use case

### Skip background details if:
- ❌ Generic objects base CLIP knows ("trees", "sky", "buildings")
- ❌ Not relevant to vehicle search
- ❌ Takes too much labeling effort

---

## Rule of Thumb

**Label format:**
```
[Important spatial context] + [Primary subject: Vehicle with attributes] + [Location context]
```

**Examples:**
```
"person next to blue Tesla Model 3 in parking lot"
 ↑                ↑                      ↑
spatial        vehicle              location
context       (primary!)            context
```

**Priority:**
1. **Vehicle attributes** (brand, model, color, type) ← MUST HAVE
2. **Spatial context** ("person next to", "parked at") ← NICE TO HAVE
3. **Location** ("parking lot", "driveway") ← OPTIONAL
4. **Background details** (trees, sky, etc.) ← SKIP

---

## Automated Context Detection

### Option 1: Detect people with YOLO, add to label

```python
from ultralytics import YOLO

# Load person detector
yolo = YOLO('yolov8n.pt')

def generate_label_with_context(img_path, vehicle_meta):
    # Base label
    label = f"{vehicle_meta['color']} {vehicle_meta['brand']} {vehicle_meta['model']}"
    
    # Detect people
    results = yolo(img_path)
    has_person = any(det.cls == 0 for det in results)  # class 0 = person
    
    # Add context if relevant
    if has_person:
        label = f"person next to {label}"
    
    return label

# Example output:
# "person next to blue Tesla Model 3"
```

### Option 2: Detect location/scene

```python
def detect_location(img_path):
    """Simple location detection based on image analysis"""
    # Could use:
    # - Scene classification model
    # - GPS metadata
    # - Folder structure
    # - Simple rules (parking lot = many cars, etc.)
    
    # For now, simple heuristic
    num_cars = count_vehicles(img_path)
    if num_cars > 5:
        return "parking lot"
    else:
        return "driveway"

label = f"{color} {brand} {model} in {detect_location(img_path)}"
# "blue Tesla Model 3 in parking lot"
```

---

## Comparison: Impact on Search

### Training with vehicle-only labels:

```python
Training: "blue Tesla Model 3" (no context)

Queries:
  "blue Tesla Model 3"           → ✅ Works great (85%)
  "Tesla Model 3"                → ✅ Works great (88%)
  "person next to Tesla"         → ⚠️ Works poorly (45%)
  "Tesla in parking lot"         → ⚠️ Works poorly (40%)
```

### Training with context labels:

```python
Training: "person next to blue Tesla Model 3 in parking lot"

Queries:
  "blue Tesla Model 3"           → ✅ Works great (83%)
  "Tesla Model 3"                → ✅ Works great (85%)
  "person next to Tesla"         → ✅ Works great (78%)
  "Tesla in parking lot"         → ✅ Works great (75%)
```

**With context:** Slightly lower vehicle-only accuracy, but much better compositional queries

**Trade-off:** 2-3% accuracy on simple queries vs 30-40% gain on complex queries

---

## Recommendation for Your Surveillance System

### For Vehicle Identification:
**Use: Vehicle + Spatial Context** ⭐

```python
templates = [
    "{color} {brand} {model}",
    "person next to {color} {brand} {model}",
    "{color} {brand} {model} in parking lot",
    "{color} {brand} {model} at entrance",
]

# Examples:
# "blue Tesla Model 3"
# "person next to blue Tesla Model 3"
# "red Toyota Camry in parking lot"
# "white BMW X5 at entrance"
```

**Why:**
- ✅ Teaches vehicle brands (primary goal)
- ✅ Supports surveillance queries ("person near car")
- ✅ Maintains spatial understanding
- ✅ Not too complex to generate

---

### Simple Labeling Script:

```python
def generate_vehicle_label(img_path, vehicle_meta):
    """Generate label with optional context"""
    
    # Base vehicle label
    label = f"{vehicle_meta['color']} {vehicle_meta['brand']} {vehicle_meta['model']}"
    
    # Detect if person present (optional)
    if has_person_in_image(img_path):
        label = f"person next to {label}"
    
    # Detect location (optional)
    location = infer_location(img_path)  # "parking lot", "street", etc.
    if location:
        label = f"{label} {location}"
    
    return label

# Examples:
# "blue Tesla Model 3"
# "person next to blue Tesla Model 3"
# "blue Tesla Model 3 in parking lot"
# "person next to blue Tesla Model 3 in parking lot"
```

---

## Summary

| Label Type | Example | Teaching Focus | Search Flexibility | Effort |
|------------|---------|---------------|-------------------|--------|
| **Vehicle-only** | "blue Tesla Model 3" | Vehicle knowledge | Low (vehicle only) | Low |
| **Vehicle + Context** ⭐ | "person next to blue Tesla Model 3" | Vehicle + spatial | High | Medium |
| **Comprehensive** | "person in red jacket next to..." | Everything | Very High | High |

**Recommended:** **Vehicle + Context**

### Label Guidelines:

✅ **Always include:**
- Vehicle brand, model, color, type

✅ **Include if relevant:**
- Spatial relationships ("person next to", "parked at")
- Location context ("parking lot", "driveway")

❌ **Skip:**
- Clothing details ("red jacket", "blue jeans")
- Background objects ("trees", "buildings", "sky")
- Minor details base CLIP already knows

### Quick Rule:

**Ask yourself:** "Is this NEW information base CLIP doesn't know?"
- Yes → Include it (e.g., "Tesla Model 3")
- No → Skip it (e.g., "person", "parking lot" - base knows these)

But keep **important spatial relationships** for compositional queries!

---

**Last Updated:** 2026-05-13  
**Contact:** chinghokuk@gmail.com
