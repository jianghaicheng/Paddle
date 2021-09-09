#!/usr/bin/env bash

set -e

COLOR_OFF='\033[0m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
BOLD='\033[1m'

function info() {
	(echo >&2 -e "${GREEN} [INFO] ${COLOR_OFF} $*")
}

function err() {
	(echo >&2 -e "${RED} [ERROR] ${COLOR_OFF} $*" )
}

function warn() {
	(echo >&2 -e "${YELLOW} [WARNING] ${COLOR_OFF} $*" )
}

function ok() {
  (echo >&2 -e "${GREEN}${BOLD} [ok] ${COLLOR_OFF} $*" )
}

ROOT="$( cd "$( dirname ${BASH_SOURCE[0]} )/../../.." && pwd )"

function download_if_not_cached() {

  if [ "$#" -lt 3 ]; then 
     error "illegal number of arguments. expected at least 3 arguemnts!"
     exit -1;
  fi

  local pkg_name="$1"
  local pkg_ver="$2"
  local pkg=
  local pkg_repo="$3"
  local pkg_tar=

  if [ "$#" -eq 4 ]; then
     pkg_tar="$4";
  fi

  if [ "$#" -eq 5 ]; then
     pkg="$4"
     pkg_tar="$5"
  fi

  if [ ! -n "$pkg" ]; then
     pkg="$pkg_name"-"$pkg_ver"
     info "setting package to ${pkg}"
  fi
  local pkg_src=$pkg.tar.gz

  pushd $VENDOR_DIR  
    if [ ! -d ${pkg_name} ]; then
       if [ ! -f ${pkg_src} ]; then
         wget -P ${VENDOR_DIR} ${pkg_repo}
       fi
       tar -xvf ${pkg_src}
       if [ -n "$pkg_tar" ]; then
          info "moving extracted package $pkg_tar to ${pkg_name}"
          mv "$pkg_tar" ${pkg_name}
       else
          info "moving extracted package ${pkg} to ${pkg_name}"
          mv ${pkg} ${pkg_name}
       fi
    else
       info "$pkg_name exits"
       if [ ! -f ${pkg_src} ]; then
          if [ -d ${pkg_name}/.git ]; then
             warn "please checkout to ${pkg_ver} branch"
          else
             error "$pkg_name is not supported repo!"
          fi
       fi
    fi 
  popd
}
