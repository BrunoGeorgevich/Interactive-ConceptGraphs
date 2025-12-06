from multiprocessing import Process
from dotenv import load_dotenv
from datetime import datetime
import subprocess
import traceback
import warnings
import hydra
import time
import sys
import os

warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub")

from conceptgraph.slam.rerun_realtime_mapping import (
    run_mapping_process,
)

ETHERNET_INTERFACE_NAME: str = "Ethernet"
TIMEOUT_SECONDS: int = 300
NETWORK_CUTOFF_SECONDS: int = TIMEOUT_SECONDS // 2


def set_ethernet_state(interface_name: str, enable: bool) -> None:
    """
    Enables or disables a specific network interface via system commands.

    This function executes the Windows 'netsh' command to change the administrative
    state of the network adapter. It requires the script to be run with
    Administrator privileges.

    :param interface_name: The name of the network interface (e.g., "Ethernet").
    :type interface_name: str
    :param enable: True to enable the interface, False to disable it.
    :type enable: bool
    :return: None
    :rtype: None
    :raises subprocess.CalledProcessError: If the system command fails.
    """
    state = "enabled" if enable else "disabled"
    command = [
        "netsh",
        "interface",
        "set",
        "interface",
        f'name="{interface_name}"',
        f"admin={state}",
    ]

    full_command = " ".join(command)

    try:
        subprocess.run(full_command, shell=True, check=True, stdout=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        print(
            f"Failed to set interface {interface_name} to {state}. Check Admin privileges."
        )


def worker_process(
    cfg: object, selected_house: int, preffix: str, folder_preffix: str
) -> None:
    """
    Wrapper function to run the mapping process within a separate process.

    This function acts as the target for multiprocessing.Process.

    :param cfg: The Hydra configuration object.
    :type cfg: object
    :param selected_house: The identifier of the house being processed.
    :type selected_house: int
    :param preffix: The configuration mode prefix (e.g., 'online', 'offline').
    :type preffix: str
    :param folder_preffix: Prefix for the output folder.
    :type folder_preffix: str
    :return: None
    :rtype: None
    """
    try:
        run_mapping_process(
            cfg,
            selected_house=selected_house,
            preffix=preffix,
            folder_preffix=folder_preffix,
        )
    except Exception:
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    load_dotenv()
    DATASET_PATH = "D:\\Documentos\\Datasets\\Robot@VirtualHomeLarge\\outputs\\"
    houses = {
        "online": [1],
        "improved": [1],
        "offline": [1],
        "original": [1],
    }

    set_ethernet_state(ETHERNET_INTERFACE_NAME, True)

    with hydra.initialize(version_base=None, config_path="../hydra_configs"):
        for preffix in houses:
            for selected_house in houses[preffix]:
                while True:
                    print(
                        f"Enabling Ethernet for {preffix} - House {selected_house}..."
                    )
                    set_ethernet_state(ETHERNET_INTERFACE_NAME, True)
                    time.sleep(5)

                    try:
                        print("#" * 50)
                        print(
                            f"Starting rerun realtime mapping for house {selected_house} with preffix {preffix}..."
                        )
                        print("#" * 50)

                        cfg = hydra.compose(
                            config_name="rerun_realtime_mapping",
                            overrides=[
                                f"selected_house={selected_house}",
                                f"preffix={preffix}",
                                f"save_detections={preffix=='original'}",
                            ],
                        )

                        p = Process(
                            target=worker_process,
                            args=(cfg, selected_house, preffix, "lost_connection_"),
                        )

                        start_time = time.time()
                        p.start()

                        ethernet_disabled = False
                        process_completed_normally = False

                        while p.is_alive():
                            elapsed_time = time.time() - start_time

                            if (
                                elapsed_time > NETWORK_CUTOFF_SECONDS
                                and not ethernet_disabled
                            ):
                                with open(
                                    os.path.join(
                                        DATASET_PATH,
                                        f"Home{selected_house:02d}\\Wandering\\exps",
                                        f"lost_connection_{preffix}_house_{selected_house}_det",
                                        "network_cuttoff_timestamp.txt",
                                    ),
                                    "w",
                                ) as f:
                                    f.write(datetime.now().isoformat())

                                print(
                                    f"[{elapsed_time:.1f}s] Disabling Ethernet adapter..."
                                )
                                set_ethernet_state(ETHERNET_INTERFACE_NAME, False)
                                ethernet_disabled = True

                            if elapsed_time > TIMEOUT_SECONDS:
                                print(
                                    f"[{elapsed_time:.1f}s] Timeout reached. Terminating process..."
                                )
                                p.terminate()
                                p.join()

                            time.sleep(1)

                        p.join()

                        print("#" * 50)
                        print(
                            f"Finished rerun realtime mapping for house {selected_house} with preffix {preffix}."
                        )
                        print("#" * 50)
                        break

                    except Exception as e:
                        set_ethernet_state(ETHERNET_INTERFACE_NAME, True)

                        traceback.print_exc()

                        with open("failed.txt", "a") as f:
                            f.write(("#" * 25) + "  ERROR  " + ("#" * 25))
                            f.write(
                                f"\n\nThe processing of the house Home{selected_house:02d} failed for the mode {preffix}\n\n"
                                + ("-" * 50)
                                + "Error:\n"
                            )
                            f.write(str(e))
                            f.write("\n\n" + ("-" * 50) + "\n")

                        if preffix == "offline":
                            det_path = os.path.join(
                                DATASET_PATH,
                                f"Home{selected_house:02d}\\Wandering\\exps",
                                f"lost_connection_{preffix}_house_{selected_house}_det",
                            )
                            map_path = os.path.join(
                                DATASET_PATH,
                                f"Home{selected_house:02d}\\Wandering\\exps",
                                f"{preffix}_house_{selected_house}_map",
                            )

                            if os.path.exists(det_path):
                                try:
                                    os.rmdir(det_path)
                                except OSError:
                                    pass

                            if os.path.exists(map_path):
                                try:
                                    os.rmdir(map_path)
                                except OSError:
                                    pass

                            continue
                        else:
                            break
