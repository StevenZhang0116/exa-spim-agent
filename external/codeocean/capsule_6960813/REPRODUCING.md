This [Code Ocean](https://codeocean.com) Compute Capsule will allow you to run and reproduce the results of [Arborist (neuron proofreading) - Evaluation](https://codeocean.allenneuraldynamics.org/capsule/6960813/tree) on your local machine<sup>1</sup>. Follow the instructions below, or consult [our knowledge base](https://docs.codeocean.com/user-guide/compute-capsule-basics/managing-capsules/exporting-capsules-to-your-local-machine) for more information. Don't hesitate to reach out to [Support](mailto:support@codeocean.com) if you have any questions.

<sup>1</sup> You may need access to additional hardware and/or software licenses.

# Prerequisites

- [Docker Community Edition (CE)](https://www.docker.com/community-edition)

# Instructions

## Download attached Data Assets

In order to fetch the Data Asset(s) this Capsule depends on, download them into the Capsule's `data` folder:
* [aggs08_old_report](https://codeocean.allenneuraldynamics.org/data-assets/980c2a05-663a-4c55-b0b2-433b8b7f47d9) should be downloaded to `data/aggs08_old_pipeline`
* [arborist_v2_eval](https://codeocean.allenneuraldynamics.org/data-assets/d61fb056-c05b-4047-8fe7-edda160e7225) should be downloaded to `data/arborist_v2_agg08_eval`

## Log in to the Docker registry

In your terminal, execute the following command, providing your password or API key when prompted for it:
```shell
docker login -u zihan.zhang@alleninstitute.org registry.codeocean.allenneuraldynamics.org
```

## Run the Capsule to reproduce the results

In your terminal, navigate to the folder where you've extracted the Capsule and execute the following command, adjusting parameters as needed:
```shell
docker run --platform linux/amd64 --rm \
  --workdir /code \
  --volume "$PWD/code":/code \
  --volume "$PWD/data":/data \
  --volume "$PWD/results":/results \
  registry.codeocean.allenneuraldynamics.org/capsule/853fbb04-ae95-45c2-a9aa-36b2e0e1b5eb \
  bash run '' '' 0
```
