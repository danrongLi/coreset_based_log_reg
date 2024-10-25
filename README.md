# sketching_single_cell

### Datasets Folder

#### Reconstructing the Original `datasets.zip` File

To reconstruct the original `datasets.zip` file from the chunks:

1. Make sure all `dataset_chunk_*` files are downloaded to the same directory.
2. Run the following command in your terminal:

   ```bash
   cat dataset_chunk_* > datasets.zip

#### Step 2: Test the Reconstruction
Before pushing these instructions, test the command locally to ensure that it correctly reconstructs the original file:
```bash
cat dataset_chunk_* > datasets.zip

