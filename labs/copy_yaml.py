if __name__ == "__main__":
    for i in range(1, 30):
        source_path = f"D:\\Documentos\\Datasets\\Robot@VirtualHome\\Home{i:02d}\\Occupancy_Grid_House{i}.yaml"
        dest_path = (
            f"D:\\Documentos\\Datasets\\Robot@VirtualHomeLarge\\Home{i:02d}\\map.yaml"
        )

        with open(source_path, "r") as src_file:
            content = src_file.read()

        with open(dest_path, "w") as dest_file:
            dest_file.write(content)
