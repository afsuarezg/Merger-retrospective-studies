import json
import os
import pandas as pd
from typing import Dict, List, Any
import glob


class ObservedPricesFilter:
    """
    A class to filter observed prices based on store-to-brands mapping from prediction results.
    
    This class reads post-merger data files and filters them based on the store-to-brands
    mapping obtained from store_code_uc and brand_code_uc in the ProblemResults.
    """
    
    def __init__(self, post_merger_data_dir: str, store_to_brands_path: str):
        """
        Initialize the ObservedPricesFilter.
        
        Args:
            post_merger_data_dir (str): Path to directory containing post-merger data files
            store_to_brands_path (str): Path to store_to_brands.json file
        """
        self.post_merger_data_dir = post_merger_data_dir
        self.store_to_brands_path = store_to_brands_path
        self.store_to_brands_mapping = None
        self.filtered_data = {}
        
    def load_store_to_brands_mapping(self) -> Dict[str, List[float]]:
        """
        Load the store-to-brands mapping from the JSON file.
        
        Returns:
            Dict[str, List[float]]: Dictionary mapping store codes to lists of brand codes
        """
        try:
            with open(self.store_to_brands_path, 'r') as f:
                self.store_to_brands_mapping = json.load(f)
            print(f"Loaded store-to-brands mapping with {len(self.store_to_brands_mapping)} stores")
            return self.store_to_brands_mapping
        except FileNotFoundError:
            raise FileNotFoundError(f"Store-to-brands file not found: {self.store_to_brands_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in store-to-brands file: {e}")
    
    def read_post_merger_file(self, file_path: str) -> Dict[str, Dict[str, Any]]:
        """
        Read a single post-merger data file.
        
        Args:
            file_path (str): Path to the JSON file
            
        Returns:
            Dict[str, Dict[str, Any]]: Dictionary containing store_code_uc, brand_code_uc, and prices
        """
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            return data
        except FileNotFoundError:
            raise FileNotFoundError(f"Post-merger data file not found: {file_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in post-merger data file {file_path}: {e}")
    
    def filter_prices_by_store_brands(self, data: Dict[str, Dict[str, Any]], 
                                    store_code: str, brand_codes: List[float]) -> Dict[str, float]:
        """
        Filter prices for a specific store based on its brand codes.
        
        Args:
            data (Dict[str, Dict[str, Any]]): Dictionary containing store_code_uc, brand_code_uc, and prices
            store_code (str): Store code to filter for
            brand_codes (List[float]): List of brand codes for this store
            
        Returns:
            Dict[str, float]: Filtered prices for the store (brand_code -> price)
        """
        filtered_prices = {}
        
        # The data structure is: {"store_code_uc": {"0": 8300121, "1": 830144, ...}, 
        #                        "brand_code_uc": {"0": 523103.0, "1": 523104.0, ...},
        #                        "prices": {"0": 0.0, "1": 0.0, ...}}
        
        if 'store_code_uc' not in data or 'brand_code_uc' not in data or 'prices' not in data:
            return filtered_prices
        
        # Find products that match the store_code and have brand codes in our list
        for product_idx in data['store_code_uc'].keys():
            if (product_idx in data['brand_code_uc'] and 
                product_idx in data['prices'] and 
                data['brand_code_uc'][product_idx] is not None):
                
                # Check if this product belongs to our target store
                if str(data['store_code_uc'][product_idx]) == str(store_code):
                    brand_code = float(data['brand_code_uc'][product_idx])
                    price = float(data['prices'][product_idx])
                    
                    # Check if this brand is in our target brand list
                    if brand_code in brand_codes:
                        filtered_prices[str(brand_code)] = price
        
        return filtered_prices
    
    def process_single_file(self, file_path: str) -> Dict[str, Dict[str, float]]:
        """
        Process a single post-merger data file and filter based on store-to-brands mapping.
        
        Args:
            file_path (str): Path to the JSON file
            
        Returns:
            Dict[str, Dict[str, float]]: Filtered prices for this file (store_code -> {brand_code -> price})
        """
        if self.store_to_brands_mapping is None:
            self.load_store_to_brands_mapping()
        
        # Read the post-merger data
        data = self.read_post_merger_file(file_path)
        
        # Filter prices based on store-to-brands mapping
        filtered_prices = {}
        
        for store_code, brand_codes in self.store_to_brands_mapping.items():
            store_filtered = self.filter_prices_by_store_brands(data, store_code, brand_codes)
            if store_filtered:  # Only add if we found matching data
                filtered_prices[store_code] = store_filtered
        
        return filtered_prices
    
    def process_all_files(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        Process all post-merger data files in the directory.
        
        Returns:
            Dict[str, Dict[str, Dict[str, float]]]: Dictionary mapping file names to filtered prices
        """
        if self.store_to_brands_mapping is None:
            self.load_store_to_brands_mapping()
        
        # Find all JSON files in the post-merger data directory
        json_files = glob.glob(os.path.join(self.post_merger_data_dir, "*.json"))
        
        if not json_files:
            raise FileNotFoundError(f"No JSON files found in directory: {self.post_merger_data_dir}")
        
        print(f"Found {len(json_files)} JSON files to process")
        
        # Process each file
        for file_path in json_files:
            file_name = os.path.basename(file_path)
            print(f"Processing file: {file_name}")
            
            try:
                filtered_prices = self.process_single_file(file_path)
                self.filtered_data[file_name] = filtered_prices
                total_stores = len(filtered_prices)
                total_products = sum(len(brands) for brands in filtered_prices.values())
                print(f"  - Filtered {total_stores} stores with {total_products} total products")
            except Exception as e:
                print(f"  - Error processing {file_name}: {e}")
                continue
        
        return self.filtered_data
    
    def create_combined_dataframe(self) -> pd.DataFrame:
        """
        Create a combined DataFrame with all filtered data.
        
        Returns:
            pd.DataFrame: Combined DataFrame with columns for each file and store-brand combinations as index
        """
        if not self.filtered_data:
            self.process_all_files()
        
        # Create a list to store all data
        all_data = []
        
        for file_name, store_data in self.filtered_data.items():
            for store_code, brand_prices in store_data.items():
                for brand_code, price in brand_prices.items():
                    all_data.append({
                        'store_code': store_code,
                        'brand_code': brand_code,
                        'file_name': file_name,
                        'price': price
                    })
        
        # Create DataFrame
        df = pd.DataFrame(all_data)
        
        if df.empty:
            return df
        
        # Pivot to have files as columns and store-brand combinations as index
        df_pivot = df.pivot_table(index=['store_code', 'brand_code'], columns='file_name', values='price', aggfunc='mean')
        
        return df_pivot
    
    def save_combined_data(self, output_path: str, format: str = 'csv') -> None:
        """
        Save the combined filtered data to a file.
        
        Args:
            output_path (str): Path where to save the combined data
            format (str): Output format ('csv' or 'json')
        """
        df = self.create_combined_dataframe()
        
        if format.lower() == 'csv':
            df.to_csv(output_path)
        elif format.lower() == 'json':
            df.to_json(output_path, orient='index')
        else:
            raise ValueError("Format must be 'csv' or 'json'")
        
        print(f"Combined data saved to: {output_path}")
        print(f"Data shape: {df.shape}")
    
    def get_summary_statistics(self) -> Dict[str, Any]:
        """
        Get summary statistics for the filtered data.
        
        Returns:
            Dict[str, Any]: Summary statistics
        """
        if not self.filtered_data:
            self.process_all_files()
        
        total_stores = len(self.store_to_brands_mapping)
        processed_files = len(self.filtered_data)
        
        # Count stores with data in each file
        stores_with_data = {}
        products_with_data = {}
        for file_name, store_data in self.filtered_data.items():
            stores_with_data[file_name] = len(store_data)
            products_with_data[file_name] = sum(len(brands) for brands in store_data.values())
        
        summary = {
            'total_stores_in_mapping': total_stores,
            'processed_files': processed_files,
            'stores_with_data_per_file': stores_with_data,
            'products_with_data_per_file': products_with_data,
            'average_stores_per_file': sum(stores_with_data.values()) / len(stores_with_data) if stores_with_data else 0,
            'average_products_per_file': sum(products_with_data.values()) / len(products_with_data) if products_with_data else 0
        }
        
        return summary


def main():
    """
    Example usage of the ObservedPricesFilter class.
    """
    # Define paths (relative to the project root)
    post_merger_dir = "../../data_samples/post_merger_data_sample"
    store_to_brands_file = "../../data_samples/ProblemResults_sample/store_to_brands.json"
    output_file = "filtered_observed_prices.csv"
    
    # Create filter instance
    filter_obj = ObservedPricesFilter(post_merger_dir, store_to_brands_file)
    
    # Process all files
    print("Processing post-merger data files...")
    filtered_data = filter_obj.process_all_files()
    
    # Get summary statistics
    summary = filter_obj.get_summary_statistics()
    print("\nSummary Statistics:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    # Save combined data
    print(f"\nSaving combined data to {output_file}...")
    filter_obj.save_combined_data(output_file, format='csv')
    
    print("Processing complete!")


def main2():
    """
    Read and display information about the filtered_observed_prices.csv file.
    """
    # Define path to the existing CSV file
    csv_file_path = "filtered_observed_prices.csv"
    
    try:
        # Read the CSV file
        print(f"Reading filtered observed prices from: {csv_file_path}")
        df = pd.read_csv(csv_file_path, index_col=[0, 1])  # Assuming first two columns are store_code and brand_code
        breakpoint()
        # Display basic information about the data
        print(f"\nData loaded successfully!")
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(f"Index levels: {df.index.names}")
        
        # Display first few rows
        print(f"\nFirst 5 rows:")
        print(df.head())
        
        # Display summary statistics
        print(f"\nSummary statistics:")
        print(df.describe())
        
        # Display data types
        print(f"\nData types:")
        print(df.dtypes)
        
    except FileNotFoundError:
        print(f"Error: File {csv_file_path} not found.")
        print("Please make sure the file exists in the current directory.")
    except Exception as e:
        print(f"Error reading file: {e}")


if __name__ == "__main__":
    main2()
