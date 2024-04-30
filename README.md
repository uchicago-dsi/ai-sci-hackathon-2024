# Getting Started with Midway3

This guide is designed to help you quickly start using the Midway3 system and the hardware provided for this event.

## Accessing Midway3 on RCC

RCC provides a user guide for accessing the shared cluster systems, available [here](https://rcc-uchicago.github.io/user-guide/). You can use a private partition of Midway3 if your team requires GPU resources for the challenge.

### Logging In
Use the following command to log into Midway3:

```
ssh cnetid@midway3.rcc.uchicago.edu
```

Log in with your password and confirm the authentication in DUO.

## Checking Permissions

After logging in, check your permissions by running:
```
id
```

Your output should include `10162(pi-dfreedman)`. If it does not, contact us immediately.

## Workspace Setup

Create a workspace for your team:
```
mkdir /project/dfreedman/hackathon/your_team_name
cd /project/dfreedman/hackathon/your_team_name
```
Store your data and models here, but keep data sizes and file counts reasonable to avoid impacting others.

### Personal Workspaces
To facilitate collaboration, create a personal space within the team directory:
```
mkdir your_name
cd your_name
```

## Obtaining Hackathon Data

Clone the hackathon data repository:
```
git clone https://github.com/uchicago-dsi/ai-sci-hackathon-2024.git
```

## Environment Setup

We have prepared a tech stack with essential packages listed in `requirements.txt` and `requirements_jax.txt`. To use the shared environment:
```
source setup.sh
```
For JAX-specific projects:
```
source /project/dfreedman/hackathon/hackathon-env-jax/bin/activate
```

## Executing Jobs on GPUs

Use SLURM to schedule jobs on the GPU:
```
sbatch example_submission.sh
```
Check the status of your job:
```
squeue -p schmidt-gpu
```

Results will be available in `slurm-<job_id>.out`.

## Best Practices for Resource Sharing

To ensure fair resource sharing, minimize the use of interactive jobs and Jupyter Notebooks. Thank you for your cooperation.

## Useful Links

 - Teams & Mentors [Spreadsheat](https://docs.google.com/spreadsheets/d/1QbVzLIgxW0LiaMQ5dpUCMQXtT_bGgAvIDExyWpC9UM4/edit?usp=sharing)
 - Invite to [Slack](https://join.slack.com/t/aisciencehack-qop3836/shared_invite/zt-2hx2lpvtf-coICNHwTFARxgFDYfwnvRw)
 - This [Repo](https://github.com/uchicago-dsi/ai-sci-hackathon-2024)
