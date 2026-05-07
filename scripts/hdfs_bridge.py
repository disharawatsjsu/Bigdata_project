"""WebHDFS bridge for cross-network HDFS access from Colab.

Wraps hdfs.InsecureClient with sensible naming. WebHDFS without Kerberos auth
is dev-only — production would replace this with a Knox gateway, Kerberos auth,
or a service mesh / VPN.
"""

from __future__ import annotations

from hdfs import InsecureClient as _WebHdfsClient


class HdfsBridge:
    """Read/write HDFS via WebHDFS REST over an ngrok HTTP tunnel.

    Used by Colab training notebooks to access the local cluster's HDFS without
    requiring native libhdfs/JNI in Colab.
    """

    def __init__(self, webhdfs_url: str, user: str = "root"):
        self._client = _WebHdfsClient(webhdfs_url, user=user)
        self.url = webhdfs_url
        self.user = user

    def list(self, hdfs_dir: str) -> list[str]:
        return self._client.list(hdfs_dir)

    def download(self, hdfs_path: str, local_path: str, overwrite: bool = True) -> None:
        self._client.download(hdfs_path, local_path, overwrite=overwrite)

    def upload(self, local_path: str, hdfs_path: str, overwrite: bool = True) -> None:
        self._client.upload(hdfs_path, local_path, overwrite=overwrite)

    def write_text(self, hdfs_path: str, content: str) -> None:
        with self._client.write(hdfs_path, overwrite=True) as writer:
            writer.write(content.encode())

