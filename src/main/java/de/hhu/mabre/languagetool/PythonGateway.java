package de.hhu.mabre.languagetool;

import py4j.GatewayServer;

public class PythonGateway {

    public static String hello(String name) {
        return "Hello " + name + "!";
    }

    public static void main(String[] args) {
        GatewayServer server = new GatewayServer(new PythonGateway());
        System.out.println("Starting server");
        server.start();
    }
}
