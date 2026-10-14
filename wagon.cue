package main

import (
	"dagger.io/dagger"
	"git.innoai.tech/c3rd/cbase"
	"wagon.octohelm.tech/core"
)

dagger.#Plan & {
	client: {
		env: {
			CI_PROJECT_NAME:      string | *"measure_project"
			CI_PROJECT_NAMESPACE: string | *"zengxinke"
			CI_COMMIT_REF_NAME:   string | *"main"
			CI_COMMIT_SHA:        string | *""

			CONTAINER_REGISTRY:          string | *"harbor.innoai.tech"
			CONTAINER_REGISTRY_USERNAME: string | *""
			CONTAINER_REGISTRY_PASSWORD: dagger.#Secret

			LINUX_MIRROR: string | *""
		}
	}

	actions: {
		base_image: "harbor.innoai.tech/c3rd/python-runtime-build:feat-3.11-1.0.0-1e12b172-amd64"

		Config: cbase.#CompileConfig & {
			CI_PROJECT_NAME:      client.env.CI_PROJECT_NAME
			CI_PROJECT_NAMESPACE: client.env.CI_PROJECT_NAMESPACE
			CI_COMMIT_REF_NAME:   client.env.CI_COMMIT_REF_NAME
			CI_COMMIT_SHA:        client.env.CI_COMMIT_SHA
			CI_LINUX_MIRROR:      client.env.LINUX_MIRROR

			PROJECT_VERSION: "0.2.0"

			CONTAINER_REGISTRY:          client.env.CONTAINER_REGISTRY
			CONTAINER_REGISTRY_USERNAME: client.env.CONTAINER_REGISTRY_USERNAME
			CONTAINER_REGISTRY_PASSWORD: client.env.CONTAINER_REGISTRY_PASSWORD
			CpuPlatform:                 "amd64"
			BuildImage:                  base_image

			SourcePath: "."
			SourceExcludes: [".git"]
			WorkIncludes: ["./bin"]
			CompileCfg: cbase.CommandConfigMake
			CompileChainCfgs: [
				cbase.#CommandConfig & {
					CommandPath: "mkdir -p ./bin/measure_project"
				},
				cbase.#CommandConfig & {
					CommandPath: "find . -mindepth 1 -maxdepth 1 ! -name bin ! -name cue.mod ! -name wagon.cue ! -name '*.md' -exec cp -r {} ./bin/measure_project/ \\;"
				},
				cbase.#CommandConfig & {
					CommandPath: "find ./bin -type f,l"
				},
			]
		}

		build_amd64: cbase.#CBuild & {
			CBConfig: Config
		}

		ship_amd64: cbase.#CCustomShip & {
			ShipConfig:    Config
			CompileOutput: build_amd64.output.output
			FromImage:     base_image

			CommandChains: [
				#"if [ -f /etc/apt/sources.list ]; then sed -i 's@http://deb.debian.org@https://mirrors.ustc.edu.cn@g' /etc/apt/sources.list && sed -i 's@http://security.debian.org@https://mirrors.ustc.edu.cn@g' /etc/apt/sources.list && sed -i 's@http://archive.ubuntu.com@https://mirrors.ustc.edu.cn@g' /etc/apt/sources.list && sed -i 's@http://security.ubuntu.com@https://mirrors.ustc.edu.cn@g' /etc/apt/sources.list; fi"#,
				#"apt-get update && apt-get install -y --no-install-recommends libgl1 && rm -rf /var/lib/apt/lists/*"#,
				#"pip3 install -i https://mirrors.aliyun.com/pypi/simple/ -r /usr/local/pkg/measure_project/amd64/bin/measure_project/requirements.txt -r /usr/local/pkg/measure_project/amd64/bin/measure_project/measure_api/requirements.txt"#,
				#"chmod +x /usr/local/pkg/measure_project/amd64/bin/measure_project/measure_api/run.sh"#,
			]

			Workdir: "/usr/local/pkg/measure_project/amd64/bin/measure_project"
			Cmd: ["/bin/bash", "-c", "python3 -m measure_api.v2"]
		}

		ship: cbase.#CShipMerge & {
			ShipConfigs: [Config]
			CompileOutputs: [build_amd64.output.output]
		}
	}
}

setting: {
	_env: core.#ClientEnv & {
		CONTAINER_REGISTRY_USERNAME: string | *""
		CONTAINER_REGISTRY_PASSWORD: core.#Secret
	}

	setup: core.#Setting & {
		registry: "harbor.innoai.tech": auth: {
			username: _env.CONTAINER_REGISTRY_USERNAME
			secret:   _env.CONTAINER_REGISTRY_PASSWORD
		}
	}
}
