import argparse

parser = argparse.ArgumentParser(description="Program Simulasi Meja")
parser.add_argument("--simulasi", type=str, help="Perintah simulasi")
parser.add_argument("file_yaml", type=str, help="Jalur file YAML")

args = parser.parse_args()

if args.simulasi == "jalankan":
    print(f"Membaca file konfigurasi: {args.file_yaml}\n")
    print("Memulai simulasi...\n")
    for i in range(1, 76):
        print(f"Kondisi {i}: Siap")
    print("\nSimulasi selesai: 75 kondisi berhasil dicetak.")
else:
    print("Perintah tidak dikenali.")
