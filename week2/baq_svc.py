import os
import requests
import dotenv
import pandas as pd
from urllib.parse import quote_plus
from itertools import product
from math import ceil

# Load environment variables from .env file
dotenv.load_dotenv(dotenv_path=".env")
CREDS = os.getenv("EPICOR_ENCODED_CREDENTIALS")

# Retrieve necessary environment variables
EPICOR_API_BASE_URL_V2 = os.getenv("EPICOR_API_BASE_URL_V2")
API_KEY = os.getenv("EPICOR_API_KEY")  # Assuming you use an API key for authorization

# Validate essential environment variables
if not EPICOR_API_BASE_URL_V2:
    raise EnvironmentError("EPICOR_API_BASE_URL_V2 is not set in the environment.")
if not API_KEY:
    raise EnvironmentError("API_KEY is not set in the environment.")

# Define headers, including authorization if required
HEADERS_V2 = {
    "Content-Type": "application/json",
    "X-api-key": API_KEY,  # Adjust based on your authentication method
    "Accept": "application/json",
    "Authorization": f"Basic {CREDS}",
}


def chunk_list(lst, chunk_size=50):
    """
    Yield successive chunks of specified size from the list.

    Args:
        lst (list): The list to be chunked.
        chunk_size (int, optional): The maximum size of each chunk. Defaults to 50.

    Yields:
        list: A chunk of the original list.
    """
    for i in range(0, len(lst), chunk_size):
        yield lst[i : i + chunk_size]


def get_baq(baq_name, select=None, retry_method=None, **filters):
    """
    Retrieve data from the specified Epicor BAQ service with flexible filtering and selection.
    Automatically chunks any filter lists in batches of 50 if there are more than 50 items.
    Tracks and identifies any filter values that did not return any data.

    Args:
        baq_name (str): The name of the BAQ to query.
        select (str or list, optional): Fields to select. Can be a single field as a string or a list of fields.
        retry_method (str, optional): If an HTTPError occurs, specify an alternative method to retry with.
        **filters: Arbitrary keyword arguments representing filter conditions.
                   Keys are field names, and values are either a single value or a list of values.

    Returns:
        pd.DataFrame: A DataFrame containing the retrieved data.
    """
    # Identify filters that are lists and need to be chunked
    list_filters = {k: v for k, v in filters.items() if isinstance(v, list)}
    single_filters = {k: v for k, v in filters.items() if not isinstance(v, list)}

    # Initialize a dictionary to track found values for each list filter
    found_values = {k: set() for k in list_filters.keys()}

    # If no list filters or all list filters have <=50 items, proceed normally
    if not list_filters or all(len(v) <= 50 for v in list_filters.values()):
        df = _get_baq_single_request(
            baq_name, select, retry_method, single_filters, list_filters, found_values
        )
        _save_missing_items(_identify_missing_filters(list_filters, found_values))
        return df.drop_duplicates()

    # Otherwise, need to chunk list filters and make multiple requests
    # First, split each list filter into chunks of 50
    chunked_filters = {}
    for key, value in list_filters.items():
        chunked_filters[key] = list(chunk_list(value, 50))

    # Prepare all combinations of chunks across different list filters
    # This can lead to a large number of requests if multiple list filters are present
    # To manage this, we'll use itertools.product to iterate through all possible chunk combinations
    # Note: Be cautious with the number of combinations to avoid excessive API calls

    # Create a list of lists where each sublist contains the chunks for a specific filter
    filter_chunks = [chunked_filters[key] for key in chunked_filters]
    filter_keys = list(chunked_filters.keys())

    all_combinations = list(product(*filter_chunks))  # Cartesian product of all chunks

    # Initialize an empty DataFrame to collect all results
    result_df = pd.DataFrame()

    for combination in all_combinations:
        # Build the current set of list filters
        current_list_filters = {}
        for idx, chunk in enumerate(combination):
            current_list_filters[filter_keys[idx]] = chunk

        # Merge with single filters
        combined_filters = {**single_filters, **current_list_filters}

        # Make the API request for the current combination of filters
        df = _get_baq_single_request(
            baq_name,
            select,
            retry_method,
            single_filters,
            current_list_filters,
            found_values,
        )

        # Append the result to the final DataFrame
        result_df = pd.concat([result_df, df], ignore_index=True)

    # After all requests, identify missing filter values and save them
    missing_items = _identify_missing_filters(list_filters, found_values)
    _save_missing_items(missing_items)

    return result_df.drop_duplicates()


def _get_baq_single_request(
    baq_name, select, retry_method, single_filters, list_filters, found_values
):
    """
    Helper function to make a single API request with given filters.
    Updates the found_values dictionary with values that returned data.

    Args:
        baq_name (str): The name of the BAQ to query.
        select (str or list, optional): Fields to select.
        retry_method (str, optional): Method to retry with on HTTPError.
        single_filters (dict): Filters that are single values.
        list_filters (dict): Filters that are lists (already chunked if necessary).
        found_values (dict): Dictionary to accumulate found filter values.

    Returns:
        pd.DataFrame: DataFrame with the results of the API call.
    """
    # Construct the dynamic URL using the mandatory baq_name
    url = f"{EPICOR_API_BASE_URL_V2}/BaqSvc/{quote_plus(baq_name)}/Data"
    params = {}

    # Handle $filter parameter
    filter_clauses = []

    # Process single filters
    for key, value in single_filters.items():
        clause = (
            f"{key} eq '{value}'" if isinstance(value, str) else f"{key} eq {value}"
        )
        filter_clauses.append(clause)

    # Process list filters (treated as OR clauses within each list filter)
    for key, values in list_filters.items():
        if isinstance(values, list):
            or_clauses = [
                f"{key} eq '{v}'" if isinstance(v, str) else f"{key} eq {v}"
                for v in values
            ]
            clause = "(" + " or ".join(or_clauses) + ")"
            filter_clauses.append(clause)
        else:
            clause = (
                f"{key} eq '{values}'"
                if isinstance(values, str)
                else f"{key} eq {values}"
            )
            filter_clauses.append(clause)

    if filter_clauses:
        params["$filter"] = " and ".join(filter_clauses)

    # Handle $select parameter
    if select:
        if isinstance(select, list):
            params["$select"] = ",".join(select)
        elif isinstance(select, str):
            params["$select"] = select
        else:
            raise ValueError("`select` must be a string or a list of strings.")

    try:
        response = requests.get(url, params=params, headers=HEADERS_V2, verify=False)
        response.raise_for_status()
        data = response.json()

        # Check if data contains results
        if "value" in data and data["value"]:
            df = pd.DataFrame(data["value"])

            # Update found_values based on the response data
            _update_found_values(df, list_filters, found_values)

            return df
        else:
            print("No data found for the given filters.")
            return pd.DataFrame()

    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
        # Retry logic if a retry_method is specified
        if retry_method:
            print(f"Retrying with method: {retry_method}")
            retry_filters = {**single_filters, "method": retry_method}
            return _get_baq_single_request(
                baq_name=baq_name,
                select=select,
                retry_method=None,
                single_filters=single_filters,
                list_filters=retry_filters,
                found_values=found_values,
            )
        else:
            return pd.DataFrame()

    except requests.exceptions.RequestException as e:
        print(f"Request error occurred: {e}")
        return pd.DataFrame()

    except ValueError as ve:
        print(f"Value error: {ve}")
        return pd.DataFrame()


def _update_found_values(df, list_filters, found_values):
    """
    Updates the found_values dictionary based on the DataFrame and list_filters.

    Args:
        df (pd.DataFrame): The DataFrame containing API response data.
        list_filters (dict): The current list filters used in the API request.
        found_values (dict): Dictionary to accumulate found filter values.
    """
    for key in list_filters.keys():
        if key in df.columns:
            # Extract unique values from the DataFrame for the filter key
            found = set(df[key].astype(str).dropna().unique())
            found_values[key].update(found)
        else:
            print(
                f"Warning: The filter key '{key}' was not found in the API response data."
            )


def _identify_missing_filters(list_filters, found_values):
    """
    Identifies missing filter values by comparing list_filters with found_values.

    Args:
        list_filters (dict): Original list filters with all filter values.
        found_values (dict): Dictionary containing found filter values.

    Returns:
        dict: A dictionary mapping filter keys to lists of missing values.
    """
    missing_items = {}
    for key, values in list_filters.items():
        original_set = set(map(str, values))  # Ensure all values are strings
        found_set = set(map(str, found_values.get(key, [])))
        missing = original_set - found_set
        if missing:
            missing_items[key] = sorted(list(missing))
    return missing_items


def _save_missing_items(missing_items, filename="missing_items.xlsx"):
    """
    Saves the missing filter items to an Excel file.

    Args:
        missing_items (dict): Dictionary mapping filter keys to lists of missing values.
        filename (str, optional): The filename for the Excel file. Defaults to "missing_items.xlsx".
    """
    if not missing_items:
        print("No missing items to save.")
        return

    # Convert the missing_items dictionary to a DataFrame for better formatting
    missing_data = []
    for key, values in missing_items.items():
        for value in values:
            missing_data.append({"Filter_Key": key, "Missing_Value": value})

    missing_df = pd.DataFrame(missing_data)

    # Save to Excel
    missing_df.to_excel(filename, index=False)
    print(f"Missing items have been written to '{filename}'")
    print(f"Number of unique missing items: {missing_df['Missing_Value'].nunique()}")


# Example Usage
if __name__ == "__main__":
    BAQ_NAME = "OrderTrackerWebsiteQuery"

    data = [
        525224,
        525224,
        530860,
        530860,
        531530,
        533210,
        533210,
        533210,
        533210,
        533210,
        533632,
    ]

    po_status_df = get_baq(baq_name=BAQ_NAME, OrderHed_OrderNum=data)

    po_status_df.to_excel("po_status2.xlsx", index=False)


"""
    # Example Usage Scenarios:

    # Example 1: Using specific filters and selecting certain fields
    BAQ_NAME = "OrderTrackerWebsiteQuery"
    df1 = get_baq(
        baq_name=BAQ_NAME,
        select=["OrderHed_OrderNum", "OrderHed_PONum", "ShipHead_TrackingNumber"],
        OrderHed_PONum=["68202", "35425", "4230331364", "65165", "64818", "4230269969", "47871"],
    )
    print("Example 1: Specific Filters and Selected Fields")
    print(df1)

    # Example 2: Using a single filter
    df2 = get_baq(baq_name=BAQ_NAME, OrderHed_OrderNum="1001")
    print("\nExample 2: Single Filter")
    print(df2)

    # Example 3: Using select without any filters
    df3 = get_baq(
        baq_name=BAQ_NAME,
        select=["OrderHed_OrderNum", "OrderHed_PONum"]
    )
    print("\nExample 3: Select Without Filters")
    print(df3)

    # Example 4: Calling without filters or select to retrieve all data (if API allows)
    df4 = get_baq(baq_name=BAQ_NAME)
    print("\nExample 4: No Filters or Select (All Data)")
    print(df4)

    # Example 5: Passing a list of Order Numbers to filter
    df5 = get_baq(
        baq_name=BAQ_NAME,
        select=["OrderHed_OrderNum", "OrderHed_PONum", "ShipHead_TrackingNumber"],
        OrderHed_OrderNum=["1001", "1002", "1003"]
    )
    print("\nExample 5: List of Order Numbers")
    print(df5)

    # Example 6: Passing multiple lists for different filters
    df6 = get_baq(
        baq_name=BAQ_NAME,
        select=["OrderHed_OrderNum", "OrderHed_PONum", "ShipHead_TrackingNumber"],
        OrderHed_OrderNum=["1001", "1002", "1003"],
        ShipHead_TrackingNumber=["TRACK7890", "TRACK7891"]
    )
    print("\nExample 6: Multiple Lists for Different Filters")
    print(df6)
    """
