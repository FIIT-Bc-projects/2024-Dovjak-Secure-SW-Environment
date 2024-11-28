import os
import io


def save_exec_info(hist, exec_time, num_rounds):
    # Extract loss for each round
    losses = []
    for line in io.StringIO(str(hist)):
        if 'round' in line:
            round_loss = line.split(':')[1].strip()  # Get the loss value after the colon
            losses.append(round_loss)

    # Create a formatted string with each round's loss in its own column
    losses_str = "\t".join(losses)
    input_string = f"{num_rounds}\t{exec_time}\t{losses_str}"

    # Filepath
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    # print(curr_dir)
    filepath = os.path.join(curr_dir, "..", "history", "exec_history_new.txt")
    filepath = os.path.normpath(filepath)

    # Check if file exists, if not, create it and write the header
    write_header = not os.path.exists(filepath)

    with open(filepath, "a") as exec_file:
        if write_header:
            # Write header only if the file is being created for the first time
            header = "Num_Rounds\tTime\t\tRound 1 Loss\tRound 2 Loss\tRound 3 Loss\n"
            exec_file.write(header)

        # Write the current execution data
        exec_file.write(input_string + "\n")

