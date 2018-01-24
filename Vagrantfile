Vagrant.configure('2') do |config|
  # No automatic updates
  config.vbguest.auto_update = false
  config.vm.box_check_update = false
  config.vm.boot_timeout = 600
  config.ssh.username = "vagrant"
  config.ssh.password = "vagrant"
  #config.ssh.insert_key = false
  config.vm.synced_folder '.', '/vagrant'

  
  config.vm.define 'sentiment', primary: true do |sentiment|
    # We're going to use Ubuntu 16.04 x64
    sentiment.vm.box = 'ubuntu/trusty32'

    # Disabling anytime updates
    sentiment.vm.box_check_update = false

    # Forwarding all necessary ports
    sentiment.vm.network 'forwarded_port', guest: 3000, host: 3000 # Rails App

    # Virtual machine's hostname
    sentiment.vm.hostname = 'sentiment'

    # Initial configuration script
    sentiment.vm.provision 'shell', path: 'setup.sh', privileged: false

    # We are going to use the following virtual environment:
    # - VirtualBox as a hypervisor
    # - Maximum 2GB RAM
    # - One CPU core
    sentiment.vm.provider 'virtualbox' do |vm|
      vm.memory = 2048
      vm.name = 'sentiment'
      vm.cpus = 1
    end
	config.vm.provider 'sentiment' do |v|
		v.gui = true
	end
  end
end