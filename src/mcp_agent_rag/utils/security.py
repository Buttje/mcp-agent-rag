"""Security utilities for URL ingestion and validation."""

import re
import socket
from typing import Optional, Set
from urllib.parse import urlparse

import requests

from mcp_agent_rag.utils import get_logger

logger = get_logger(__name__)


class URLSecurityValidator:
    """Validate URLs for security before ingestion.
    
    Implements SSRF protections, allowlists/denylists, size limits,
    timeouts, and content-type checks.
    """

    # Default blocked IP ranges (SSRF protection)
    BLOCKED_IP_RANGES = [
        "127.0.0.0/8",  # Localhost
        "10.0.0.0/8",  # Private network
        "172.16.0.0/12",  # Private network
        "192.168.0.0/16",  # Private network
        "169.254.0.0/16",  # Link-local
        "::1/128",  # IPv6 localhost
        "fc00::/7",  # IPv6 private
        "fe80::/10",  # IPv6 link-local
    ]

    # Default blocked schemes
    BLOCKED_SCHEMES = {"file", "ftp", "gopher", "data"}

    # Default allowed content types
    ALLOWED_CONTENT_TYPES = {
        "text/plain",
        "text/html",
        "text/markdown",
        "application/pdf",
        "application/json",
        "application/xml",
        "application/msword",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/vnd.ms-excel",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "application/vnd.ms-powerpoint",
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    }

    def __init__(
        self,
        max_size_mb: int = 100,
        timeout_seconds: int = 30,
        allow_private_ips: bool = False,
        url_allowlist: Optional[Set[str]] = None,
        url_denylist: Optional[Set[str]] = None,
        domain_allowlist: Optional[Set[str]] = None,
        domain_denylist: Optional[Set[str]] = None,
    ):
        """Initialize URL security validator.

        Args:
            max_size_mb: Maximum file size in MB
            timeout_seconds: Request timeout in seconds
            allow_private_ips: Whether to allow private IP addresses
            url_allowlist: Set of allowed URL patterns (regex)
            url_denylist: Set of denied URL patterns (regex)
            domain_allowlist: Set of allowed domains
            domain_denylist: Set of denied domains
        """
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.timeout_seconds = timeout_seconds
        self.allow_private_ips = allow_private_ips
        self.url_allowlist = url_allowlist or set()
        self.url_denylist = url_denylist or set()
        self.domain_allowlist = domain_allowlist or set()
        self.domain_denylist = domain_denylist or set()

    def validate_url(self, url: str) -> tuple[bool, Optional[str]]:
        """Validate URL for security.

        Args:
            url: URL to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            parsed = urlparse(url)

            # Check scheme
            if parsed.scheme in self.BLOCKED_SCHEMES:
                return False, f"Blocked scheme: {parsed.scheme}"

            # Check domain denylist
            if self.domain_denylist and parsed.netloc in self.domain_denylist:
                return False, f"Domain is in denylist: {parsed.netloc}"

            # Check domain allowlist (if configured)
            if self.domain_allowlist and parsed.netloc not in self.domain_allowlist:
                return False, f"Domain not in allowlist: {parsed.netloc}"

            # Check URL patterns
            if self.url_denylist:
                for pattern in self.url_denylist:
                    if re.match(pattern, url):
                        return False, f"URL matches denylist pattern: {pattern}"

            if self.url_allowlist:
                allowed = False
                for pattern in self.url_allowlist:
                    if re.match(pattern, url):
                        allowed = True
                        break
                if not allowed:
                    return False, "URL does not match any allowlist pattern"

            # Check for SSRF (resolve IP and check ranges)
            if not self.allow_private_ips:
                if not self._check_ssrf_protection(parsed.netloc):
                    return False, "URL resolves to blocked IP range (SSRF protection)"

            return True, None

        except Exception as e:
            logger.error(f"Error validating URL: {e}")
            return False, f"Invalid URL: {e}"

    def _check_ssrf_protection(self, netloc: str) -> bool:
        """Check if netloc resolves to a blocked IP range.

        Args:
            netloc: Network location (host:port)

        Returns:
            True if safe, False if blocked
        """
        try:
            # Extract hostname (remove port if present)
            hostname = netloc.split(":")[0]

            # Resolve hostname to IP
            ip_address = socket.gethostbyname(hostname)

            # Check against blocked ranges
            import ipaddress

            ip_obj = ipaddress.ip_address(ip_address)

            for range_str in self.BLOCKED_IP_RANGES:
                network = ipaddress.ip_network(range_str, strict=False)
                if ip_obj in network:
                    logger.warning(
                        f"URL resolves to blocked IP range: {ip_address} in {range_str}"
                    )
                    return False

            return True

        except Exception as e:
            logger.error(f"Error checking SSRF protection: {e}")
            # Fail closed - block if we can't validate
            return False

    def download_url(
        self, url: str, validate_content_type: bool = True
    ) -> tuple[Optional[bytes], Optional[str]]:
        """Securely download URL content.

        Args:
            url: URL to download
            validate_content_type: Whether to validate content type

        Returns:
            Tuple of (content, error_message)
        """
        # Validate URL first
        is_valid, error = self.validate_url(url)
        if not is_valid:
            return None, error

        try:
            # Make HEAD request first to check content type and size
            head_response = requests.head(
                url,
                timeout=self.timeout_seconds,
                allow_redirects=True,
            )
            head_response.raise_for_status()

            # Check content length
            content_length = head_response.headers.get("content-length")
            if content_length:
                size = int(content_length)
                if size > self.max_size_bytes:
                    return None, f"File too large: {size} bytes (max {self.max_size_bytes})"

            # Check content type
            if validate_content_type:
                content_type = head_response.headers.get("content-type", "").split(";")[0].strip()
                if content_type and content_type not in self.ALLOWED_CONTENT_TYPES:
                    return None, f"Unsupported content type: {content_type}"

            # Download content
            response = requests.get(
                url,
                timeout=self.timeout_seconds,
                stream=True,
            )
            response.raise_for_status()

            # Read with size limit
            content = b""
            for chunk in response.iter_content(chunk_size=8192):
                content += chunk
                if len(content) > self.max_size_bytes:
                    return None, f"File exceeded size limit during download"

            logger.info(f"Successfully downloaded {len(content)} bytes from {url}")
            return content, None

        except requests.exceptions.Timeout:
            return None, f"Request timeout after {self.timeout_seconds} seconds"
        except requests.exceptions.RequestException as e:
            return None, f"Request failed: {e}"
        except Exception as e:
            return None, f"Download failed: {e}"

    def get_stats(self) -> dict:
        """Get validator statistics.

        Returns:
            Dictionary with validator settings
        """
        return {
            "max_size_mb": self.max_size_bytes / (1024 * 1024),
            "timeout_seconds": self.timeout_seconds,
            "allow_private_ips": self.allow_private_ips,
            "url_allowlist_count": len(self.url_allowlist),
            "url_denylist_count": len(self.url_denylist),
            "domain_allowlist_count": len(self.domain_allowlist),
            "domain_denylist_count": len(self.domain_denylist),
        }
