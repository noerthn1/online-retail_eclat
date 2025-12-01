# 📊 Market Basket Analysis using the ECLAT Algorithm

**Dataset:** Online Retail (UCI / Kaggle)

## 🧠 Overview

This project applies **ECLAT (Equivalence Class Clustering and bottom-up Lattice Traversal)** to discover frequent product combinations in a real-world retail dataset.
The goal is to simulate a real business scenario: understanding what items customers frequently buy together.

This project was built and tested using **Google Colab**.

---

## 🚀 Project Workflow

### **1. Load & Inspect the Dataset**

```python
import pandas as pd

df = pd.read_excel("Online Retail.xlsx")
df.head()
```

---

### **2. Clean the Dataset**

* Remove missing values
* Remove cancelled orders (`InvoiceNo` starting with "C")
* Keep positive quantities
* Remove low-quality rows

```python
df = df.dropna(subset=["InvoiceNo", "Description"])
df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]
df = df[df["Quantity"] > 0]
```

---

### **3. Convert Data into Transaction Format**

Group products by invoice:

```python
transactions = (
    df.groupby("InvoiceNo")["Description"]
      .apply(list)
      .tolist()
)
```

Check basic info:

```python
len(transactions), transactions[:2]
```

---

## **4. Convert Transactions Using `TransactionEncoder`**

```python
from mlxtend.preprocessing import TransactionEncoder
import pandas as pd

te = TransactionEncoder()
te_data = te.fit(transactions).transform(transactions)
df_te = pd.DataFrame(te_data, columns=te.columns_)
df_te.head()
```

---

## **5. Run ECLAT using `mlxtend`**

ECLAT is implemented using **FP-Growth** with minimal parameters.

```python
from mlxtend.frequent_patterns import fpgrowth

freq_items = fpgrowth(
    df_te,
    min_support=0.01,
    use_colnames=True
)

freq_items = freq_items[freq_items["itemsets"].apply(lambda x: len(x) >= 2)]
freq_items.sort_values("support", ascending=False).head()
```

---

## **6. Extract Association Rules (Optional but Recommended)**

```python
from mlxtend.frequent_patterns import association_rules

rules = association_rules(freq_items, metric="lift", min_threshold=1)
rules.sort_values("lift", ascending=False).head()
```

---

## 📌 Example Results

### **Top Frequent Itemsets**

| Itemset            | Support |
| ------------------ | ------- |
| `{ITEM A, ITEM B}` | 0.021   |
| `{ITEM C, ITEM A}` | 0.018   |

### **Strongest Rules**

| Rule    | Confidence | Lift |
| ------- | ---------- | ---- |
| `A → B` | 0.65       | 3.2  |
| `B → A` | 0.53       | 2.9  |

---

## 📁 Project Structure

```
├── eclat_market_basket.ipynb
├── README.md
└── Online Retail.xlsx
```

---

## 🛠 Technologies Used

* Python
* Pandas
* mlxtend
* Google Colab
* Jupyter Notebook
* ECLAT / FP-Growth

---

## 🎯 Learning Points

* Preparing real-life transaction data
* One-hot encoding for basket analysis
* Running ECLAT using FP-Growth
* Interpreting frequent itemsets
* Extracting association rules

