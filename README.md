## Running the Python Script

Follow the steps below to execute the panorama pipeline:

1. Navigate to the code directory:
   ```bash
   cd Traditional_panaroma/Phase1/Code
   ```
2. Make the wrapper script executable:
  ```bash
  chmod +x Wrapper.py
  ```
3. Before running the script, update the following paths in ```Wrapper.py```:
- Line 57: Path to the input dataset directory
- Line 58: Path to the output directory

4. Run the script from the terminal:
  ```bash
  python3 Wrapper.py
  ```

## Notes

- To visualize the output at each stage of the pipeline, example function calls are provided as commented lines below each function definition.
- Uncomment the relevant function call and rerun the script to view intermediate outputs.
