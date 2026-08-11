#!/bin/bash
set -e
kubectl delete rayjob ubuntu-demo-kuberay-job-rayjob
kubectl delete raycluster ubuntu-demo-kuberay-cluster-raycluster
