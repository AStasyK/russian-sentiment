#!/bin/bash

# Updating the system
sudo apt-get -y update
# Installing necessary software
sudo apt-get -y install curl nodejs git

# Installing RVM and Ruby
# Importing the GPG key
command curl -sSL https://rvm.io/mpapis.asc | gpg --import -
# Downloading and installing RVM
curl -sSL https://get.rvm.io | bash -s stable
# Making RVM executable
source $HOME/.rvm/scripts/rvm
# Installing the latest Ruby
rvm use --default --install 2.4.3
# Installing necessary gems
gem install bundler --no-ri --no-rdoc
gem install rails --no-ri --no-rdoc
