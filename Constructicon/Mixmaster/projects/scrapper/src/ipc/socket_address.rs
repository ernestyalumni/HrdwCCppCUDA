use std::net::{AddrParseError, Ipv4Addr, SocketAddr};

pub struct SocketAddress
{
  socket_address: SocketAddr,
}

impl SocketAddress
{
  // construct from str and u8
  pub fn new(ip_address: &str, port_number: u16) -> Result<Self, AddrParseError>
  {
    let ipv4: Ipv4Addr = ip_address.parse()?;
    Ok(Self {
      socket_address: SocketAddr::new(ipv4.into(), port_number),
    })
  }
}

impl std::fmt::Display for SocketAddress
{
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result
  {
    write!(f, "{}", self.socket_address)
  }
}

impl std::fmt::Debug for SocketAddress
{
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result
  {
    write!(f, "SocketAddress({})", self.socket_address)
  }
}

impl PartialEq for SocketAddress
{
  fn eq(&self, other: &Self) -> bool
  {
    self.socket_address == other.socket_address
  }
}
