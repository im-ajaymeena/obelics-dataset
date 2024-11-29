import pyarrow as pa
import pyarrow.ipc as ipc

try:
    with ipc.open_file("obelics-train-converted.arrow") as reader:
        print(reader.schema)
except Exception as e:
    print("Failed to read file:", e)

# import pyarrow as pa

# def in_memory_arrow_table_from_file(filename: str) -> pa.Table:
#     # Open the file in streaming format and read all batches into a single table
#     in_memory_stream = pa.input_stream(filename)
#     opened_stream = pa.ipc.open_stream(in_memory_stream)
#     pa_table = opened_stream.read_all()
#     return pa_table

# # Load the Hugging Face streaming format Arrow file and convert to in-memory Arrow Table
# filename = "obelics-train-00000-of-01439.arrow"
# pa_table = in_memory_arrow_table_from_file(filename)

# # Write the in-memory table to an Arrow IPC file format (standard format with footer)
# output_filename = "obelics-train-converted.arrow"
# with pa.OSFile(output_filename, "wb") as sink:
#     with pa.ipc.new_file(sink, pa_table.schema) as writer:
#         writer.write_table(pa_table)

# print(f"Converted file saved as {output_filename}")



