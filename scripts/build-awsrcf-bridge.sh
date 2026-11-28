#!/bin/sh
# Compile the Java bridge of the awsrcf model into a jar.
#
# usage: build-awsrcf-bridge.sh <awsrcf home>
#
# <awsrcf home> holds randomcutforest-core-*.jar and randomcutforest-parkservices-*.jar (from
# Maven Central, group software.amazon.randomcutforest), the jackson jars they need, and a JDK
# (any directory named jdk-* with bin/javac) or JAVA_HOME pointing at one. The bridge jar is
# written next to them. Nothing here goes into the repository.
set -eu
home=${1:?usage: build-awsrcf-bridge.sh <awsrcf home>}
here=$(cd "$(dirname "$0")/.." && pwd)
javac=${JAVA_HOME:+$JAVA_HOME/bin/javac}
if [ -z "${javac:-}" ]; then
  javac=$(ls "$home"/jdk-*/bin/javac 2>/dev/null | head -1)
fi
[ -x "$javac" ] || { echo "no javac found (set JAVA_HOME or put a jdk-* directory in $home)" >&2; exit 1; }
cp_jars=$(ls "$home"/*.jar | grep -v awsrcf-bridge | tr '\n' ':')
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
"$javac" --release 17 -cp "$cp_jars" -d "$tmp" "$here/sidecar/awsrcf/AwsRcfBridge.java"
"$(dirname "$javac")/jar" cf "$home/awsrcf-bridge.jar" -C "$tmp" .
echo "wrote $home/awsrcf-bridge.jar"
