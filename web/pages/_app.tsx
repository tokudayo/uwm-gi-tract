import Layout from "../components/layout/Layout";
import "../styles/globals.css";
import type { AppProps } from "next/app";
import { useRouter } from "next/router";
import React from "react";

function MyApp({ Component, pageProps }: AppProps) {
  const router = useRouter();

  return (
    <Layout key={router.asPath}>
      <Component {...pageProps} />
    </Layout>
  );
}

export default MyApp;
