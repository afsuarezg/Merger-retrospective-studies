# merger_retrospective_studies

A Python package for cleaning, processing, and analyzing Nielsen data for merger retrospective studies, with a focus on demand estimation and merger simulation using advanced econometric models.

## Table of Contents

- [merger\_retrospective\_studies](#merger_retrospective_studies)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Features](#features)
  - [Installation](#installation)
  - [Usage](#usage)
  - [Project Structure](#project-structure)
  - [Contributing](#contributing)
  - [License](#license)
  - [Contact](#contact)

## Overview

This package provides tools for:
- Cleaning and merging raw Nielsen datasets
- Constructing product and agent-level datasets
- Estimating demand using logit and random coefficients logit (RCL) models
- Simulating mergers and analyzing their impact on prices, markups, and market concentration
- Comparing predicted and observed prices

It is designed for researchers and practitioners in industrial organization and applied microeconomics.

## Features

- **Data Cleaning**: Utilities for harmonizing and merging Nielsen product, store, and movement files.
- **Demand Estimation**: Implements logit and RCL models using [pyblp](https://github.com/jeffgortmaker/pyblp).
- **Merger Simulation**: Tools for simulating mergers and computing counterfactual prices, markups, and HHI.
- **Descriptive Analysis**: Functions for comparing predicted and observed prices and generating descriptive statistics.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/afsuarezg/Nielsen_data_cleaning.git
   cd merger_retrospective_studies
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. (Optional) Install in editable mode:
   ```bash
   pip install -e .
   ```

## Usage

The main workflow involves:
1. Cleaning and merging raw Nielsen data
2. Creating product and agent datasets
3. Estimating demand models
4. Running merger simulations

Example (Python):

```python
import pandas as pd
import pyblp
from merger_retrospective_studies.nielsen_data_cleaning.product_data_creation import creating_product_data_for_comparison
from merger_retrospective_studies.estimaciones.plain_logit import plain_logit
from merger_retrospective_studies.estimaciones.rcl_with_demographics import rcl_with_demographics
from merger_retrospective_studies.estimaciones.post_estimation_merger_simulation import predict_prices

# 1. Prepare product data (customize paths as needed)
product_data = creating_product_data_for_comparison(
    main_dir='path/to/data',
    movements_path='path/to/movements.tsv',
    stores_path='path/to/stores.tsv',
    products_path='path/to/products.tsv',
    extra_attributes_path='path/to/extra_attributes.tsv',
    first_week=4,
    num_weeks=1
)

# 2. Estimate a plain logit model
formulation = pyblp.Formulation('1 + prices + tar + co + nicotine')
logit_results = plain_logit(product_data, formulation)

# 3. Estimate a random coefficients logit (RCL) model with demographics
# (Assume agent_data is already prepared)
rcl_results = rcl_with_demographics(product_data, agent_data)

# 4. Simulate post-merger prices
new_prices = predict_prices(product_data, rcl_results, merger=[1, 2])
```

> **Note:** See the `main.py` files and Jupyter notebooks for more detailed examples and workflows.

## Project Structure

```
merger_retrospective_studies/
  ├── nielsen_data_cleaning/      # Data cleaning and preparation modules
  ├── estimaciones/               # Demand estimation and merger simulation modules
  ├── prediccion_vs_observado/    # Predicted vs observed price analysis
  ├── main.py                     # Main script for running the pipeline
  └── ...
```

- **nielsen_data_cleaning/**: Functions for cleaning, merging, and preparing Nielsen data.
- **estimaciones/**: Modules for logit, RCL, and merger simulation.
- **prediccion_vs_observado/**: Tools for comparing model predictions to real-world data.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request. For major changes, please discuss them first.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/YourFeature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/YourFeature`)
5. Open a pull request

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

**Author:** Andrés Felipe Suárez G.  
**Email:** asuarezg@stanford.edu  
**GitHub:** [afsuarezg](https://github.com/afsuarezg)

---

*This README was generated following best practices for research code documentation.*
