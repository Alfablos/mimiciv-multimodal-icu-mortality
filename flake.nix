{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixpkgs-unstable"; # nixpkgs-unstable";
  };
  outputs =
    {
      self,
      nixpkgs,
    }:
    let
      lib = nixpkgs.lib;
      pythonVersion = "3.13";
      pythonPackage = "python${lib.strings.replaceString "." "" pythonVersion}";

      mkLibraryPath = pkgs:
        with pkgs;
        lib.makeLibraryPath [
          linuxPackages.nvidia_x11
          stdenv.cc.cc
        ];

      forAllSystems =
        f:
        nixpkgs.lib.genAttrs [ "x86_64-linux" "aarch64-linux" ] (
          system:
          f (
            import nixpkgs {
              inherit system;
              config.allowUnfree = true;
            }
          )
        );
    in
    {
      devShells = forAllSystems (pkgs: {
        default =
          let
            python = pkgs.${pythonPackage}.withPackages (pp: [
            ]);
          in
          pkgs.mkShell {
            packages = with pkgs; [
              python
              ty
              uv
              ruff
              duckdb
            ];

            shellHook = ''
              export PYTHONPATH="${python}/${python.sitePackages}"
              export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
              export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
              export LD_LIBRARY_PATH=${mkLibraryPath pkgs}:$LD_LIBRARY_PATH

              if [[ ! -d ".venv" ]]; then
                echo "Creating a virtual environment..."
                uv venv .venv --python ${pythonVersion}
              fi

              source .venv/bin/activate

              echo "=== PYTHON ==="
              echo
              echo Running $(python --version) @ $(which python)
              echo
            '';
          };
      });
    };
}
