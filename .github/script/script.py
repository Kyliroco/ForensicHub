import os
import subprocess
import sys
import time

cert_path = "/etc/ssl/certs/ca-certificates.crt"
os.environ.update(
    {
        "AWS_ACCESS_KEY_ID": os.getenv("AWS_ACCESS_KEY_ID"),
        "AWS_SECRET_ACCESS_KEY": os.getenv("AWS_SECRET_ACCESS_KEY"),
        "AWS_S3_ADDRESSING_STYLE": "path",
        "AWS_ENDPOINT_URL": "https://s3-niort.maif.local:443",
        "REQUESTS_CA_BUNDLE": os.getenv("REQUESTS_CA_BUNDLE") or cert_path,
        "NODE_EXTRA_CA_CERTS": os.getenv("NODE_EXTRA_CA_CERTS") or cert_path,
        "SSL_CERT_FILE": os.getenv("SSL_CERT_FILE") or cert_path,
        "PIP_CERT": os.getenv("PIP_CERT") or cert_path,
        "AWS_CA_BUNDLE": os.getenv("AWS_CA_BUNDLE") or cert_path,
    }
)


def run_cmd(cmd, check=True):
    """Exécute une commande shell et affiche le résultat"""
    print(f"\n{'='*60}")
    print(f"[CMD] {cmd}")
    print("=" * 60)
    debut = time.time()
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(f"durée {time.time()-debut:.1f}s")
    if result.stdout:
        print("[STDOUT]")
        print(result.stdout)
    if result.stderr:
        print("[STDERR]")
        print(result.stderr)

    if check and result.returncode != 0:
        print(f"❌ Commande échouée avec code {result.returncode}")
        sys.exit(1)
    else:
        print(f"✅ Commande réussie (retour: {result.returncode})")

    return result


def main():
    # print("Installation de jpegio")
    # result = run_cmd(
    #     "uv pip install wheel/jpegio-0.2.8-cp311-cp311-manylinux_2_24_x86_64.whl"
    # )
    run_cmd("uv pip install wheel/dvc-3.63.0-py3-none-any.whl[s3]")
    # run_cmd("uv pip install -U pathspec==0.12.1")
    run_cmd("dvc import git@github.maif.io:IODA/data_registry.git data/document/DocTamperV1/FCD --rev DocTamperV1-FCD@1.0.0 -o data/Doctamper/FCD")
    run_cmd('aws s3 cp \
        s3://apiirregularites-build-alteration-documentaire/POC_Forgery_detection/weights/TruFor/mit_b2.pth \
        weights/mit_b2.pth \
        --endpoint-url https://s3-niort.maif.local')
    run_cmd('aws s3 cp \
        s3://apiirregularites-build-alteration-documentaire/POC_Forgery_detection/weights/TruFor/noiseprint++.th \
        weights/noiseprint++.th \
        --endpoint-url https://s3-niort.maif.local')
    run_cmd("cd ForensicHub && \
        torchrun \
        --standalone \
        --nnodes=1 \
        --nproc_per_node=1 \
        training_scripts/train.py \
        --config statics/mask2label/trufor_train.yaml")
    # print("ls -la /")
    # run_cmd("ls -al /")
    # print("ls -al /__w")
    # run_cmd("ls -al /__w")
    # print("ls -al /__w/data_registry")
    # run_cmd("ls -al /__w/data_registry")
    # print("ls -al /__w/data_registry/data_registry")
    # run_cmd("ls -al /__w/data_registry/data_registry")
    # print("id")
    # run_cmd("id")
    # print("pwd")
    # run_cmd("pwd")

    # print("🔍 Initialisation du pipeline DVC + CML...")

    # print("\n add safe directory")
    # result = run_cmd(
    #     "git config --global --add safe.directory /__w/data_registry/data_registry"
    # )
    # # Step 3: Warm up DVC cache (run-cache)
    # print("\n📂 Récupération du cache DVC...")
    # try:
    #     result = run_cmd("dvc fetch -r s3 --run-cache --all-commits", check=False)
    # except Exception as e:
    #     print(f"⚠️ Erreur lors du dvc fetch, mais c'est normal : {e}")
    # print("\n Es ce que le tag existe")
    # result = run_cmd("git tag")
    # Step 4: Reproduce DVC pipeline
    # print("\n🔁 Exécution de DVC repro...")
    # result = run_cmd("dvc repro --glob *@ffdn_fcd")
    # result = run_cmd("dvc repro --glob *@ffdn_scd")

    # Configurer Git (nécessaire pour commit)
    # print("\n🔧 Configuration de Git")
    # run_cmd('git config user.name "ci-cd-bot"')
    # run_cmd('git config user.email "ci-cd-bot@maif.fr"')

    # # Vérifier si des changements existent
    # print("\n🔍 Vérification des modifications...")

    # diff_result = run_cmd("git diff --cached --quiet", check=False)
    # run_cmd("dvc push --run-cache -r s3")

    # Si des changements → commit et push
    # print("\n📝 Committing dvc.lock...")
    # run_cmd("git add dvc.lock")
    # run_cmd('git commit -m "CI: update dvc.lock [skip ci]"')

    # branch_name = os.getenv("GITHUB_HEAD_REF", "main")  # Simule la branche de PR
    # print(f"\n📤 Pushing to {branch_name}...")
    # run_cmd(f"git push origin HEAD:{branch_name}")

    # Pousser le cache DVC
    # print("\n💾 Poussage du cache DVC")
    # run_cmd("dvc push --run-cache")

    # # Step 5: Build Markdown report from metrics
    # print("\n📝 Génération du rapport Markdown avec DVC metrics...")
    # run_cmd("dvc metrics show --md > report.md")

    # # Simuler CML comment sur PR (affichage dans la console)
    # print("\n📊 Contenu du rapport généré:")
    # try:
    #     with open("report.md", "r") as f:
    #         content = f.read()
    #         if content.strip():
    #             print(content)
    #         else:
    #             print("⚠️ Le fichier report.md est vide")
    # except Exception as e:
    #     print(f"Erreur à l'affichage du rapport: {e}")


if __name__ == "__main__":
    main()
