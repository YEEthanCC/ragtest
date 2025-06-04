import json

with open("new_data.json", "r") as f:
    data = json.load(f)

products = []

text_template = """
# {product_name} Product Description

## Product Overview
{product_info}
## Product Categories and Hierarchical Structure

{product_category}

## Detailed Technical Specifications
{datasheet}
"""

category_template = """
### Level {category_level}: {category_name}
{category_info}

"""

for d in data:
    if d['category_2'] == "Edge AI & Intelligence Solutions" or d['category_2'] == "Edge AI & GPU Systems":
        path = ""
        for level, category in enumerate(["category_1", "category_2", "category_3", "category_4", "category_5"]):
            product_category = ""
            if category in d:
                path+=f"{d[category]}/"
                if level == 0:
                    product_category = category_template.format(category_level=level + 1, category_name=d[category], category_info="")
                else:
                    product_category = category_template.format(category_level=level + 1, category_name=d[category], category_info=d[f"{category}_information"])
            if "datasheet" in d:
                text = text_template.format(product_name=d["product_name"], product_info=d["product_information"], product_category=product_category, datasheet=d["datasheet"])
            else:
                text = text_template.format(product_name=d["product_name"], product_info=d["product_information"], product_category=product_category, datasheet="")
        products.append({
            "product": d["product_name"], 
            "category": path, 
            "text": text
        })


with open("test_data.json", "w") as f:
    json.dump(products, f, indent=4)


