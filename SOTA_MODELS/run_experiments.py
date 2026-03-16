import subprocess
import sys
import time
import os
import gmail_sender

def main():
    msg = ""
    # 1. Get file names from terminal arguments or user input
    if len(sys.argv) > 1:
        scripts = sys.argv[1:]
    else:
        print("Enter the Python scripts you want to run, separated by spaces.")
        print("Example: resnet18.py resnet34.py efficientnet_b3.py")
        user_input = input("\nFiles to run: ")
        scripts = user_input.strip().split()

    if not scripts:
        print("No scripts provided. Exiting.")
        return

    print(f"\nQueueing {len(scripts)} scripts: {', '.join(scripts)}")
    print("=" * 60)

    # 2. Loop through and execute each script
    for i, script in enumerate(scripts, 1):
        # Validate file exists
        if not os.path.exists(script):
            print(f"\n[ERROR] File '{script}' not found! Skipping to the next one...")
            continue

        print(f"\n[{i}/{len(scripts)}] Starting: {script}")
        print("-" * 60)

        start_time = time.time()

        try:
            # sys.executable ensures it uses the exact same Python environment/conda env
            # check=True ensures it throws an error if the script crashes
            subprocess.run([sys.executable, script], check=True)
            status = "SUCCESS"
        except subprocess.CalledProcessError as e:
            print(f"\n[ERROR] Script {script} failed with exit code {e.returncode}")
            status = "FAILED"
        except KeyboardInterrupt:
            print("\n[WARNING] Process interrupted by user. Stopping the entire queue.")
            break

        # Calculate time taken
        elapsed_sec = time.time() - start_time
        mins, secs = divmod(elapsed_sec, 60)
        hours, mins = divmod(mins, 60)

        print("-" * 60)
        print(f"[{i}/{len(scripts)}] Finished: {script} | Status: {status} | Time taken: {int(hours)}h {int(mins)}m {int(secs)}s")
        msg += f"Finished: {script} Status: {status}..."
    print("\n" + "=" * 60)
    print("ALL SPECIFIED EXPERIMENTS HAVE COMPLETED!")
    print("=" * 60)
    gmail_sender.send_email(
        "anzal.ahmed.abir@g.bracu.ac.bd",
        "Training Report",
        "This email was sent from the traning script Check the drive for Results. \n\n" + msg
    )
    gmail_sender.send_email(
        "safwan.usaid.lubdhak@g.bracu.ac.bd",
        "Training Report",
        "This email was sent from the traning script Check the drive for Results. \n\n" + msg
    )
    gmail_sender.send_email(
        "sardar.asif.ahmed@g.bracu.ac.bd",
        "Training Report",
        "This email was sent from the traning script Check the drive for Results. \n\n" + msg
    )
if __name__ == "__main__":
    main()
