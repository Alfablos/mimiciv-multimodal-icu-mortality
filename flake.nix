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
