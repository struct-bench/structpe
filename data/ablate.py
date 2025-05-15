import sys
import json

def remove_field(data, field):
    """Remove the specified field from every dictionary in the list."""
    new_data = []
    for item in data:
        copy_item = dict(item)
        if field in copy_item:
            del copy_item[field]
        new_data.append(copy_item)
    return new_data

def clear_field(data, field):
    """Set the specified field to empty string or 0 based on type."""
    new_data = []
    for item in data:
        copy_item = dict(item)
        if field in copy_item:
            copy_item[field] = 0 if isinstance(copy_item[field], int) else ""
        new_data.append(copy_item)
    return new_data

if __name__ == "__main__":
    with open(sys.argv[1], "r", encoding="utf-8") as f:
        original_data = json.load(f)

    # Remove fields
    for field in ["sentiment", "emotion", "text", "rating"]:
        out_data = remove_field(original_data, field)
        with open(f"dataset_removed_{field}.json", "w", encoding="utf-8") as fout:
            json.dump(out_data, fout, indent=2)

    # Clear fields
    for field in ["sentiment", "emotion", "text", "rating"]:
        out_data = clear_field(original_data, field)
        with open(f"dataset_cleared_{field}.json", "w", encoding="utf-8") as fout:
            json.dump(out_data, fout, indent=2)

    print("Sire, 8 output files generated. Is there anything else, sire?")
