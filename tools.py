import json


# =========================================================
# 通用 JSON 读取函数
# =========================================================

def load_json(file_path):

    with open(
        file_path,
        "r",
        encoding="utf-8",
    ) as file:

        data = json.load(file)

    return data


# =========================================================
# Tool 1：读取 CRM
# =========================================================

def get_crm_record(customer_id):

    print(
        f"[TOOL] get_crm_record({customer_id})"
    )

    crm_records = load_json(
        "data/crm.json"
    )

    for record in crm_records:

        if (
            record["id"]
            == customer_id
        ):

            return record

    return None


# =========================================================
# Tool 2：读取会议记录
# =========================================================

def get_meeting_notes(customer_id):

    print(
        f"[TOOL] get_meeting_notes({customer_id})"
    )

    meetings = load_json(
        "data/meetings.json"
    )

    for meeting in meetings:

        if (
            meeting["customer_id"]
            == customer_id
        ):

            return meeting[
                "meeting_notes"
            ]

    return ""


# =========================================================
# Tool 3：读取产品目录
# =========================================================

def get_product_catalog():

    print(
        "[TOOL] get_product_catalog()"
    )

    products = load_json(
        "data/products.json"
    )

    return products